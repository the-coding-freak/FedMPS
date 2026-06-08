import torch
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class TelemetryController:
    def __init__(self, args, logdir, classes_list):
        self.args = args
        self.logdir = logdir
        self.classes_list = classes_list
        self.num_users = args.num_users
        
        # State tracking
        self.prev_local_high_protos = {}  # {client_id: {class_id: tensor}}
        self.prev_local_low_protos = {}   # {client_id: {class_id: tensor}}
        self.staleness = {idx: {c: 0 for c in classes_list[idx]} for idx in range(self.num_users)}
        
        # Logs list to be saved to CSV
        self.drift_records = []
        self.byte_records = []
        
    def log_round(self, round_idx, local_high_protos, local_low_protos, global_high_protos, global_low_protos, global_logits):
        # 1. Cosine drift calculation
        # Combine drifts using: D_{i,c,t} = \alpha D^h_{i,c,t} + \beta D^l_{i,c,t}
        # alpha = 0.7, beta = 0.3
        alpha = 0.7
        beta = 0.3
        
        # Save high and low prototypes for verification by round, client, class
        proto_save_dir = os.path.join(self.logdir, 'prototypes')
        os.makedirs(proto_save_dir, exist_ok=True)
        
        saved_protos = {}
        total_uplink_bytes = 0
        total_downlink_bytes = 0
        
        for idx in range(self.num_users):
            saved_protos[idx] = {}
            for c in self.classes_list[idx]:
                h_proto = local_high_protos[idx][c]
                l_proto = local_low_protos[idx][c]
                
                # Detach and CPU-convert for storage
                h_proto_cpu = h_proto.detach().cpu()
                l_proto_cpu = l_proto.detach().cpu()
                saved_protos[idx][c] = {
                    'high': h_proto_cpu,
                    'low': l_proto_cpu
                }
                
                # Compute drift
                d_high = 0.0
                d_low = 0.0
                
                if idx in self.prev_local_high_protos and c in self.prev_local_high_protos[idx]:
                    h_prev = self.prev_local_high_protos[idx][c]
                    l_prev = self.prev_local_low_protos[idx][c]
                    
                    # F.cosine_similarity expects inputs of at least 1D, so check shape
                    cos_high = F.cosine_similarity(h_proto_cpu.unsqueeze(0), h_prev.unsqueeze(0)).item()
                    cos_low = F.cosine_similarity(l_proto_cpu.unsqueeze(0), l_prev.unsqueeze(0)).item()
                    
                    # Cosine drift: 1 - cos(a, b)
                    d_high = 1.0 - cos_high
                    d_low = 1.0 - cos_low
                
                d_combined = alpha * d_high + beta * d_low
                
                # Record drift
                self.drift_records.append({
                    'round': round_idx,
                    'client_id': idx,
                    'class_id': int(c),
                    'drift_high': float(d_high),
                    'drift_low': float(d_low),
                    'drift_combined': float(d_combined),
                    'staleness': int(self.staleness[idx][c])
                })
                
                # Update previous prototypes state
                if idx not in self.prev_local_high_protos:
                    self.prev_local_high_protos[idx] = {}
                    self.prev_local_low_protos[idx] = {}
                self.prev_local_high_protos[idx][c] = h_proto_cpu
                self.prev_local_low_protos[idx][c] = l_proto_cpu
                
                # 2. Byte count calculations
                # Uplink: client sends local prototypes to server
                h_bytes = h_proto.numel() * 4
                l_bytes = l_proto.numel() * 4
                total_uplink_bytes += (h_bytes + l_bytes)
                
                # Downlink: server sends global prototypes and soft labels to client
                if c in global_high_protos:
                    gh_bytes = global_high_protos[c][0].numel() * 4
                    gl_bytes = global_low_protos[c][0].numel() * 4
                    total_downlink_bytes += (gh_bytes + gl_bytes)
                if len(global_logits) > 0 and c in global_logits:
                    g_logit_bytes = global_logits[c].numel() * 4
                    total_downlink_bytes += g_logit_bytes
            
            # Since everything is synchronized every round in Week 2, staleness remains 0
            # (In Week 3, if skipped, we will increment staleness)
            
        # Save prototypes for this round
        torch.save(saved_protos, os.path.join(proto_save_dir, f'round_{round_idx}.pt'))
        
        # Record bytes
        self.byte_records.append({
            'round': round_idx,
            'uplink_bytes': total_uplink_bytes,
            'downlink_bytes': total_downlink_bytes,
            'total_bytes': total_uplink_bytes + total_downlink_bytes
        })
        
    def save_logs(self):
        # Save CSVs
        drift_df = pd.DataFrame(self.drift_records)
        drift_csv_path = os.path.join(self.logdir, 'drift_logs.csv')
        drift_df.to_csv(drift_csv_path, index=False)
        
        byte_df = pd.DataFrame(self.byte_records)
        byte_csv_path = os.path.join(self.logdir, 'prototype_payload_logs.csv')
        byte_df.to_csv(byte_csv_path, index=False)
        
        print(f"Saved drift logs to {drift_csv_path}")
        print(f"Saved payload logs to {byte_csv_path}")
        
        # Generate plot of average drift per round
        if len(drift_df) > 0:
            avg_drift_per_round = drift_df.groupby('round')['drift_combined'].mean().reset_index()
            plt.figure(figsize=(8, 5))
            plt.plot(avg_drift_per_round['round'], avg_drift_per_round['drift_combined'], marker='o', color='b', label='Average Combined Drift')
            plt.xlabel('Communication Round')
            plt.ylabel('Average Semantic Drift')
            plt.title('Average Prototype Semantic Drift per Round')
            plt.grid(True)
            plt.legend()
            
            plot_path = os.path.join(self.logdir, 'average_drift_per_round.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Generated drift plot at {plot_path}")
