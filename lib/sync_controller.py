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
        
        # Week 4 QoS and MEC parameters
        self.sync_trace_file = getattr(args, 'sync_trace_file', None)
        self.sync_qos_lambda = getattr(args, 'sync_qos_lambda', 1.0)
        self.sync_qos_mu = getattr(args, 'sync_qos_mu', 1.0)
        self.sync_mec_aggregation = getattr(args, 'sync_mec_aggregation', 0)
        
        # Load network trace file
        if self.sync_trace_file is not None and os.path.exists(self.sync_trace_file):
            self.trace_df = pd.read_csv(self.sync_trace_file)
            print(f"Loaded network trace from {self.sync_trace_file}")
        else:
            self.trace_df = None
            if self.sync_trace_file is not None:
                print(f"Warning: Network trace file {self.sync_trace_file} not found. Using ideal network mode.")
        
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
                drift_data[(idx, c_int)] = (d_high, d_low, d_combined)
                
                # Update previous prototypes state (always freshly trained)
                if idx not in self.prev_local_high_protos:
                    self.prev_local_high_protos[idx] = {}
                    self.prev_local_low_protos[idx] = {}
                self.prev_local_high_protos[idx][c_int] = h_proto_cpu
                self.prev_local_low_protos[idx][c_int] = l_proto_cpu
                
        # 2. QoS Cost calculation and Score scaling
        client_to_mec = {idx: 0 for idx in range(self.num_users)}
        has_qos_data = False
        if self.trace_df is not None:
            round_rows = self.trace_df[self.trace_df['round'] == round_idx]
            if not round_rows.empty:
                qos_by_client = {}
                for _, row in round_rows.iterrows():
                    cid = int(row['client_id'])
                    qos_by_client[cid] = {
                        'mec_id': int(row.get('mec_id', 0)),
                        'uplink_mbps': float(row.get('uplink_mbps', 100.0)),
                        'latency_ms': float(row.get('latency_ms', 0.0)),
                        'packet_loss': float(row.get('packet_loss', 0.0))
                    }
                has_qos_data = True
                
        client_qos_raw = {}
        client_class_pairs = []
        for idx in range(self.num_users):
            for c in self.classes_list[idx]:
                client_class_pairs.append((idx, int(c)))
                
        for idx in range(self.num_users):
            if has_qos_data and idx in qos_by_client:
                mec_id = qos_by_client[idx]['mec_id']
                uplink_mbps = max(qos_by_client[idx]['uplink_mbps'], 1e-3)
                latency_ms = qos_by_client[idx]['latency_ms']
                packet_loss = qos_by_client[idx]['packet_loss']
            else:
                mec_id = 0
                uplink_mbps = 100.0
                latency_ms = 0.0
                packet_loss = 0.0
                
            client_to_mec[idx] = mec_id
            
            # Compute transmission delay dynamically
            c_first = int(self.classes_list[idx][0])
            h_proto = local_high_protos[idx][c_first]
            l_proto = local_low_protos[idx][c_first]
            proto_megabits = (h_proto.numel() + l_proto.numel()) * 32 / 1000000.0
            delay_sec = proto_megabits / uplink_mbps
            
            client_qos_raw[idx] = {
                'delay': delay_sec,
                'latency': latency_ms,
                'loss': packet_loss
            }
            
        # Min-Max Normalization across clients in the round
        delays_arr = np.array([client_qos_raw[idx]['delay'] for idx, _ in client_class_pairs])
        latencies_arr = np.array([client_qos_raw[idx]['latency'] for idx, _ in client_class_pairs])
        losses_arr = np.array([client_qos_raw[idx]['loss'] for idx, _ in client_class_pairs])
        
        def min_max_scale(arr):
            amin = arr.min()
            amax = arr.max()
            diff = amax - amin
            if diff < 1e-8:
                return np.zeros_like(arr)
            return (arr - amin) / diff
            
        norm_delays = min_max_scale(delays_arr)
        norm_latencies = min_max_scale(latencies_arr)
        norm_losses = min_max_scale(losses_arr)
        
        qos_costs = {}
        for pair_idx, (idx, c_int) in enumerate(client_class_pairs):
            c_cost = norm_delays[pair_idx] + self.sync_qos_lambda * norm_latencies[pair_idx] + self.sync_qos_mu * norm_losses[pair_idx]
            qos_costs[(idx, c_int)] = c_cost
            
        # Calculate Priority Scores
        for idx, c_int in client_class_pairs:
            d_high, d_low, d_combined = drift_data[(idx, c_int)]
            if self.trace_df is not None:
                score = (d_combined + self.sync_rho * (self.r_staleness[idx][c_int] / self.sync_staleness_K)) / (1.0 + qos_costs[(idx, c_int)])
            else:
                score = d_combined + self.sync_rho * (self.r_staleness[idx][c_int] / self.sync_staleness_K)
            scores[(idx, c_int)] = score
        
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
                    'score': float(score),
                    'qos_cost': float(qos_costs[(idx, c_int)])
                })
                
                # Update cache and staleness
                if is_sync:
                    self.cached_local_high_protos[idx][c_int] = local_high_protos[idx][c_int].clone()
                    self.cached_local_low_protos[idx][c_int] = local_low_protos[idx][c_int].clone()
                    self.r_staleness[idx][c_int] = 0
                else:
                    self.r_staleness[idx][c_int] += 1
                    
        # 3. Byte count calculations based on sync decisions
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
            
        # 4. Compute MEC-to-cloud backhaul bytes
        total_mec_to_cloud_bytes = 0
        if self.sync_mec_aggregation == 1:
            mec_classes = {}
            for idx in range(self.num_users):
                mec_id = client_to_mec[idx]
                if mec_id not in mec_classes:
                    mec_classes[mec_id] = set()
                for c in self.classes_list[idx]:
                    c_int = int(c)
                    if sync_decisions[idx][c_int]:
                        mec_classes[mec_id].add(c_int)
                        
            for mec_id, classes_set in mec_classes.items():
                for c_int in classes_set:
                    # Find a client under this MEC that synced this class to get its size
                    sample_client = None
                    for idx in range(self.num_users):
                        if client_to_mec[idx] == mec_id and c_int in self.classes_list[idx] and sync_decisions[idx][c_int]:
                            sample_client = idx
                            break
                    if sample_client is not None:
                        h_proto = local_high_protos[sample_client][c_int]
                        l_proto = local_low_protos[sample_client][c_int]
                        h_bytes = h_proto.numel() * 4
                        l_bytes = l_proto.numel() * 4
                        total_mec_to_cloud_bytes += (h_bytes + l_bytes)
        else:
            total_mec_to_cloud_bytes = total_uplink_bytes

        # Save prototypes for this round
        torch.save(saved_protos, os.path.join(proto_save_dir, f'round_{round_idx}.pt'))
        
        # Record bytes, skipped percentage and MEC-to-cloud backhaul bytes
        skipped_pct = (num_skipped / total_pairs) * 100.0 if total_pairs > 0 else 0.0
        self.byte_records.append({
            'round': round_idx,
            'uplink_bytes': total_uplink_bytes,
            'downlink_bytes': total_downlink_bytes,
            'total_bytes': total_uplink_bytes + total_downlink_bytes,
            'skipped_percentage': skipped_pct,
            'mec_to_cloud_bytes': total_mec_to_cloud_bytes
        })
        
        return self.cached_local_high_protos, self.cached_local_low_protos, sync_decisions, client_to_mec
        
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
