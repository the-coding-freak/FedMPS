import argparse
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic 5G network traces for FedMPS simulations.")
    parser.add_argument('--output', type=str, default='exps/synthetic_trace.csv', help="output path for CSV file")
    parser.add_argument('--rounds', type=int, default=10, help="number of rounds to simulate")
    parser.add_argument('--num_users', type=int, default=5, help="number of users (clients)")
    parser.add_argument('--scenario', type=str, default='bottleneck', choices=['ideal', 'bottleneck', 'unstable'],
                        help="QoS scenario type")
    parser.add_argument('--num_mecs', type=int, default=2, help="number of MEC nodes")
    args = parser.parse_args()

    records = []
    
    # Assign clients to MEC nodes in a round-robin fashion
    client_mecs = {idx: (idx % args.num_mecs) for idx in range(args.num_users)}

    for r in range(args.rounds):
        for idx in range(args.num_users):
            mec_id = client_mecs[idx]
            
            if args.scenario == 'ideal':
                # Infinite-like rate, 0 latency, 0 loss
                uplink_mbps = 100.0
                latency_ms = 0.0
                packet_loss = 0.0
            elif args.scenario == 'bottleneck':
                # Clients 0 and 1 have constrained uplink, others have high-speed links
                if idx in [0, 1]:
                    uplink_mbps = 1.0  # 1 Mbps bottleneck
                    latency_ms = 100.0 # higher latency
                    packet_loss = 0.05
                else:
                    uplink_mbps = 50.0
                    latency_ms = 10.0
                    packet_loss = 0.00
            elif args.scenario == 'unstable':
                # Low bandwidth, high latency, high packet loss for everyone
                uplink_mbps = 2.0
                latency_ms = 250.0
                packet_loss = 0.15
                
            records.append({
                'round': r,
                'client_id': idx,
                'mec_id': mec_id,
                'uplink_mbps': uplink_mbps,
                'latency_ms': latency_ms,
                'packet_loss': packet_loss
            })

    df = pd.DataFrame(records)
    df.to_csv(args.output, index=False)
    print(f"Successfully generated synthetic 5G trace ({args.scenario} scenario) for {args.num_users} users over {args.rounds} rounds at {args.output}")

if __name__ == '__main__':
    main()
