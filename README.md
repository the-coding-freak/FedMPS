# 5G-FedMPS: Extending FedMPS with 5G-NR Telemetry & Edge MEC Aggregation

> **Reference Baseline Paper**: This repository is built upon and extends the foundational work:
> **FedMPS**: W. Yang, X. Hu, X. Zhu, R. Wu, W. Pedrycz, X. Liu, and J. Huang, *"FedMPS: Federated Learning in a Synergy of Multi-Level Prototype-Based Contrastive Learning and Soft Label Generation,"* **IEEE Transactions on Neural Networks and Learning Systems (TNNLS)**, 2025. DOI: [`10.1109/TNNLS.2025.3611832`](https://doi.org/10.1109/TNNLS.2025.3611832).

---

## ⚡ Quickstart & Reproducibility Guide

### 1. Requirements & Dependencies
On Kaggle or Google Colab environments, install `tensorboardX`:
```bash
pip install tensorboardX
# Or install via requirements file:
pip install -r requirements.txt
```

### 2. Step 1: Generate 5G-LENA NR Telemetry Trace (ns-3 C++)
To re-generate the 500-round 5G New Radio network telemetry trace (20 mobile UEs, 3.5 GHz Sub-6, 100 MHz CC, 3GPP A2-A4 handovers across 3 MEC servers):
```bash
# From your ns-3.38 workspace directory:
./ns3 run "fedmps_5glena_sim"
```
*Outputs*: `exps/cifar_5glena_trace.csv` (9,980 rows covering 20 UEs $\times$ 499 rounds; Round 0 is a forced full-sync initialization round).

### 3. Step 2: Run 5G-FedMPS Training (PyTorch)
Navigate to the `exps/` directory before running training commands. All experiments use a fixed seed (`--seed 1234`) for deterministic reproduction on CIFAR-10 Non-IID (3-way, 100-shot, 20 clients, 500 rounds):

```bash
cd exps
```

* **Main Proposed 5G-FedMPS (QoS Gating + MEC Aggregation)**:
```bash
python federated_main.py --alg ours --dataset cifar10 --num_classes 10 --num_users 20 --ways 3 --shots 100 --rounds 500 --seed 1234
```
*Expected Result*: Peak Test Accuracy **82.75%** (Round 312), suppressing prototype payload volume by **50.8%--98.3%** and reducing core backhaul traffic by >50%.

* **Ablation 1: QoS-Only Scheduling (No Semantic Drift Tracking)**:
```bash
python federated_main.py --alg ours --dataset cifar10 --num_classes 10 --num_users 20 --ways 3 --shots 100 --rounds 500 --seed 1234 --sync_qos_only 1
```
*Expected Result*: Peak Test Accuracy **82.36%** (Round 269), demonstrating that representation drift tracking is indispensable.

* **Ablation 2: No-MEC Edge Aggregation (Direct Central Cloud Backhaul)**:
```bash
python federated_main.py --alg ours --dataset cifar10 --num_classes 10 --num_users 20 --ways 3 --shots 100 --rounds 500 --seed 1234 --sync_mec_aggregation 0
```
*Expected Result*: Peak Test Accuracy **83.28%** (Round 367), but forces **100% of client prototype volume over central core backhaul** (increasing traffic from ~14.5 KB to ~60.3 KB/round).

---

## 🔗 Executable Kaggle Experiment Notebooks

For full transparency and instant cloud reproduction, all experimental milestones, 5G-LENA trace generation, and ablations are hosted on Kaggle:

0. **5G-LENA Trace Generator (ns-3 C++)**: [5g-lena-trace-generation](https://www.kaggle.com/code/silversraileigh/5g-lena-trace-generation)
1. **Div 1 (Baseline FedMPS)**: [div-1-final](https://www.kaggle.com/code/silversraileigh/div-1-final)
2. **Div 2 (Bottleneck Network FedMPS)**: [div-2-final](https://www.kaggle.com/code/silversraileigh/div-2-final)
3. **Div 3 (Fixed Drift Gating, $\delta=0.01$)**: [div-3-final-1](https://www.kaggle.com/code/silversraileigh/div-3-final-1)
4. **Div 3 (Adaptive Drift Gating, $\gamma=0.5$)**: [div-3-final-2](https://www.kaggle.com/code/silversraileigh/div-3-final-2)
5. **Div 4 (5G-FedMPS Main Proposed Method)**: [div-4-final-5glena](https://www.kaggle.com/code/silversraileigh/div-4-final-5glena)
6. **Div 5 (Ablation 1: QoS-Only Scheduling)**: [div-5-final-1-qos-only](https://www.kaggle.com/code/silversraileigh/div-5-final-1-qos-only)
7. **Div 5 (Ablation 2: No-MEC Edge Aggregation)**: [div-5-final-2-mec](https://www.kaggle.com/code/silversraileigh/div-5-final-2-mec)

---

## 📖 Baseline Citation
To cite the foundational FedMPS paper:
```bibtex
@ARTICLE{FedMPS,
  author={Yang, Wenxin and Hu, Xingchen and Zhu, Xiubin and Wu, Rouwan and Pedrycz, Witold and Liu, Xinwang and Huang, Jincai},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={FedMPS: Federated Learning in a Synergy of Multi-Level Prototype-Based Contrastive Learning and Soft Label Generation}, 
  year={2025},
  doi={10.1109/TNNLS.2025.3611832}}
```
