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
        
        # Week 3 synchronization parameters
        self.sync_threshold_type = getattr(args, 'sync_threshold_type', 'fixed')
        self.sync_threshold = getattr(args, 'sync_threshold', 0.01)
        self.sync_gamma = getattr(args, 'sync_gamma', 0.5)
        self.sync_staleness_K = getattr(args, 'sync_staleness_K', 10)
        self.sync_rho = getattr(args, 'sync_rho', 0.0)
        
        # State tracking
        self.prev_local_high_protos = {}  # {client_id: {class_id: tensor}} (fresh local from prev round)
        self.prev_local_low_protos = {}   # {client_id: {class_id: tensor}}
        
        self.r_staleness = {idx: {int(c): 0 for c in classes_list[idx]} for idx in range(self.num_users)}
        self.cached_local_high_protos = {idx: {} for idx in range(self.num_users)}
        self.cached_local_low_protos = {idx: {} for idx in range(self.num_users)}
        
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
        scores = {}
        drift_data = {}
        
        for idx in range(self.num_users):
            saved_protos[idx] = {}
            for c in self.classes_list[idx]:
                c_int = int(c)
                h_proto = local_high_protos[idx][c_int]
                l_proto = local_low_protos[idx][c_int]
                
                # Detach and CPU-convert for storage
                h_proto_cpu = h_proto.detach().cpu()
                l_proto_cpu = l_proto.detach().cpu()
                saved_protos[idx][c_int] = {
                    'high': h_proto_cpu,
                    'low': l_proto_cpu
                }
                
                # Compute drift
                d_high = 0.0
                d_low = 0.0
                
                if idx in self.prev_local_high_protos and c_int in self.prev_local_high_protos[idx]:
                    h_prev = self.prev_local_high_protos[idx][c_int]
                    l_prev = self.prev_local_low_protos[idx][c_int]
                    
                    cos_high = F.cosine_similarity(h_proto_cpu.unsqueeze(0), h_prev.unsqueeze(0)).item()
                    cos_low = F.cosine_similarity(l_proto_cpu.unsqueeze(0), l_prev.unsqueeze(0)).item()
                    
                    d_high = 1.0 - cos_high
                    d_low = 1.0 - cos_low
                
                d_combined = alpha * d_high + beta * d_low
                
                # Priority Score = D_{i,c,t} + \rho * (r_{i,c,t} / K)
                score = d_combined + self.sync_rho * (self.r_staleness[idx][c_int] / self.sync_staleness_K)
                scores[(idx, c_int)] = score
                drift_data[(idx, c_int)] = (d_high, d_low, d_combined)
                
                # Update previous prototypes state (always freshly trained)
                if idx not in self.prev_local_high_protos:
                    self.prev_local_high_protos[idx] = {}
                    self.prev_local_low_protos[idx] = {}
                self.prev_local_high_protos[idx][c_int] = h_proto_cpu
                self.prev_local_low_protos[idx][c_int] = l_proto_cpu
        
        # Determine the gating threshold tau_t
        if self.sync_threshold_type == 'fixed':
            threshold = self.sync_threshold
        elif self.sync_threshold_type == 'adaptive':
            if round_idx == 0:
                threshold = 0.0
            else:
                all_scores = list(scores.values())
                if len(all_scores) > 0:
                    median_score = np.median(all_scores)
                    mad_score = np.median(np.abs(np.array(all_scores) - median_score))
                    threshold = median_score + self.sync_gamma * mad_score
                else:
                    threshold = 0.0
        else:
            threshold = 0.0

        # Determine sync decisions and update caches / staleness
        sync_decisions = {idx: {} for idx in range(self.num_users)}
        num_skipped = 0
        total_pairs = 0
        
        for idx in range(self.num_users):
            for c in self.classes_list[idx]:
                c_int = int(c)
                total_pairs += 1
                score = scores[(idx, c_int)]
                staleness = self.r_staleness[idx][c_int]
                d_high, d_low, d_combined = drift_data[(idx, c_int)]
                
                is_sync = (round_idx == 0) or (self.sync_threshold_type == 'fixed' and self.sync_threshold <= 0.0) or (score > threshold) or (staleness >= self.sync_staleness_K)
                sync_decisions[idx][c_int] = is_sync
                
                if not is_sync:
                    num_skipped += 1
                
                # Append to drift records
                self.drift_records.append({
                    'round': round_idx,
                    'client_id': idx,
                    'class_id': c_int,
                    'drift_high': float(d_high),
                    'drift_low': float(d_low),
                    'drift_combined': float(d_combined),
                    'staleness': int(staleness),
                    'synced': bool(is_sync),
                    'score': float(score)
                })
                
                # Update cache and staleness
                if is_sync:
                    self.cached_local_high_protos[idx][c_int] = local_high_protos[idx][c_int].clone()
                    self.cached_local_low_protos[idx][c_int] = local_low_protos[idx][c_int].clone()
                    self.r_staleness[idx][c_int] = 0
                else:
                    self.r_staleness[idx][c_int] += 1
                    
        # 2. Byte count calculations based on sync decisions
        total_uplink_bytes = 0
        total_downlink_bytes = 0
        
        for idx in range(self.num_users):
            for c in self.classes_list[idx]:
                c_int = int(c)
                if sync_decisions[idx][c_int]:
                    # Uplink
                    h_proto = local_high_protos[idx][c_int]
                    l_proto = local_low_protos[idx][c_int]
                    h_bytes = h_proto.numel() * 4
                    l_bytes = l_proto.numel() * 4
                    total_uplink_bytes += (h_bytes + l_bytes)
                    
                    # Downlink
                    if c_int in global_high_protos:
                        gh_bytes = global_high_protos[c_int][0].numel() * 4
                        gl_bytes = global_low_protos[c_int][0].numel() * 4
                        total_downlink_bytes += (gh_bytes + gl_bytes)
                    if len(global_logits) > 0 and c_int in global_logits:
                        g_logit_bytes = global_logits[c_int].numel() * 4
                        total_downlink_bytes += g_logit_bytes
            
        # Save prototypes for this round
        torch.save(saved_protos, os.path.join(proto_save_dir, f'round_{round_idx}.pt'))
        
        # Record bytes and skipped percentage
        skipped_pct = (num_skipped / total_pairs) * 100.0 if total_pairs > 0 else 0.0
        self.byte_records.append({
            'round': round_idx,
            'uplink_bytes': total_uplink_bytes,
            'downlink_bytes': total_downlink_bytes,
            'total_bytes': total_uplink_bytes + total_downlink_bytes,
            'skipped_percentage': skipped_pct
        })
        
        return self.cached_local_high_protos, self.cached_local_low_protos, sync_decisions
        
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
