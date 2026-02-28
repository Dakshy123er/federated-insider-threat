#  Federated Insider Threat Detection — CERT v5.2

Binary classification of insider threats using **pFedMe** federated learning with per-user personalisation and a BiLSTM-Attention architecture. Raw employee data never leaves the department — only model weights are shared.

---

## Method overview

| Stage | What happens |
|---|---|
| **Stage 1** | Each department trains locally for 5 epochs per round. Only the shared body (BiLSTM-Attention encoder) is aggregated on a central server via FedAvg. The proximal term (λ=0.5) prevents client drift. |
| **Stage 2** | The global body is frozen. Each user fine-tunes their own head and deviation encoder on their personal history. |

Key design choices:

- **Binary labels** — Normal (0) vs Insider (1), much more stable than multi-class
- **Focal loss** γ=1.5 — gently focuses on hard examples without destroying precision
- **2× minority oversampling** — conservative; avoids recall collapse
- **Precision-first threshold** — threshold selected so precision ≥ 0.5 on validation
- **Deviation encoder** — captures how *this specific user* deviates from their own norm
- **F-beta (β=0.5) early stopping** — rewards precision twice as much as recall

---

## Project structure

```
.
├── insider_threat_detection_humanized.py   # main script
├── requirements.txt
└── README.md
```

---

## Quick start

```bash
# 1. Clone the repo
git clone https://github.com//federated-insider-threat.git
cd federated-insider-threat

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place the dataset in the project root
#    Expected: week-r5.2.csv  or  week-r5.2.csv.gz

# 4. Run (Google Colab recommended for GPU)
python insider_threat_detection_humanized.py
```

---

## Dataset

[CERT Insider Threat Dataset v5.2](https://resources.sei.cmu.edu/library/asset-view.cfm?assetid=508099) — Carnegie Mellon University SEI.  
OR
Access From here (https://drive.google.com/file/d/1SbB1fiBdBn21Rs7Uu5kFKuPcyNhWTBL9/view?usp=sharing)

---

## Output files

| File | Description |
|---|---|
| `eda_plots.png` | Class distributions, temporal activity patterns |
| `training_curves.png` | Stage-1 loss, F1, AUC across FL rounds |
| `threshold_analysis.png` | PR curve and metric sweeps vs threshold |
| `binary_results.png` | Confusion matrix, ROC, PR, per-dept F1 |
| `baseline_comparison.png` | Head-to-head bar charts |
| `ablation_study.png` | Per-component contribution |
| `privacy_utility.png` | Privacy-utility frontier |
| `results_summary.xlsx` | All tables in 8 sheets |
| `best_stage1_checkpoint.pt` | Best federated body weights |
| `best_stage2_checkpoint.pt` | Personalised user model weights |

---

## Requirements

- Python ≥ 3.9
- CUDA-capable GPU strongly recommended (tested on NVIDIA A100)
- See `requirements.txt` for full package list
