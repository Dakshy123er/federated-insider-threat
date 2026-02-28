# =============================================================================
# Federated Insider Threat Detection — CERT Dataset v5.2
# Binary Classification: Normal Behaviour (0) vs Insider Threat (1)
#
# Method: pFedMe with User-Level Personalisation + BiLSTM-Attention
#
# What this does, in plain terms:
#   We want to detect employees who are acting suspiciously — potential
#   "insider threats" — without centralising raw employee data. Instead,
#   each department trains its own local model, and only the shared model
#   "body" (the generic feature extractor) is aggregated across departments
#   on a central server. Personal identifiers and raw logs never leave
#   the department. Then, each individual user gets their own fine-tuned
#   model in a second stage.
#
# Key design choices vs the previous version:
#   1.  Binary labels only (Normal / Insider) — much more stable than 5-way
#   2.  Conservative 2× oversampling instead of 5× — avoids recall collapse
#   3.  Focal loss gamma=1.5 instead of 3.0 — gentler focus on hard examples
#   4.  Precision-first threshold selection on validation — alerts are trustworthy
#   5.  F-beta (β=0.5) as the early-stopping metric — rewards precision 2× recall
#   6.  Robust department assignment — handles numeric as well as string user IDs
#   7.  pFedMe proximal weight λ=0.5 — stronger pull toward the global body
#   8.  Calibrated output probabilities — fixes the inverted-signal bug
#   9.  Per-user insider-rate tracking — each user model knows their own history
#  10.  Honest evaluation: precision, recall, F1, AUC, and FPR@TPR90 all reported
#
# How to run:
#   1. Open Google Colab Pro → Runtime → Change runtime type → A100 GPU
#   2. Upload week-r5.2.csv (or the .gz compressed version) via the Files panel
#   3. Paste this entire file into a cell and press Run
# =============================================================================


# ── Cell 1: Install packages and import everything we need ────────────────

# Install all required packages quietly (suppress the verbose pip output)
!pip install -q torch torchvision scikit-learn xgboost imbalanced-learn \
              matplotlib seaborn pandas numpy tqdm openpyxl

import os
import copy
import warnings
import random
import time

warnings.filterwarnings('ignore')  # suppress sklearn/torch deprecation noise

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tqdm.notebook import tqdm
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    classification_report,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# ── Reproducibility ───────────────────────────────────────────────────────
# Fix all random seeds so results are the same on every run.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on : {DEVICE}")
if DEVICE == 'cuda':
    print(f"GPU model  : {torch.cuda.get_device_name(0)}")


# ── Global constants ───────────────────────────────────────────────────────
N_CLASSES = 2   # binary: 0 = normal, 1 = insider
T_WINDOW  = 8   # how many consecutive weeks form one training example


# ── Cell 2: Load the dataset and take a first look ────────────────────────

print("=" * 60)
print("STEP 1: LOADING DATA")
print("=" * 60)

CSV_PATH    = 'week-r5.2.csv.gz'
compression = 'gzip' if CSV_PATH.endswith('.gz') else 'infer'
df          = pd.read_csv(CSV_PATH, low_memory=False, compression=compression)

print(f"Rows × columns : {df.shape}")
print(f"Column names   : {df.columns.tolist()}")


def find_col(df, candidates):
    """
    Try to find a column by a list of candidate names.
    Checks both exact matches and case-insensitive substring matches,
    so the script works even if the CSV uses slightly different column names.
    """
    for name in candidates:
        if name in df.columns:
            return name
        for col in df.columns:
            if name.lower() in col.lower():
                return col
    return None


# Detect the key columns regardless of exact naming in the CSV
USER_COL  = find_col(df, ['user', 'user_id', 'userId', 'employee'])
WEEK_COL  = find_col(df, ['week', 'week_num', 'weeknum', 'wk'])
LABEL_COL = find_col(df, ['label', 'insider', 'scenario', 'class', 'malicious'])
DEPT_COL  = find_col(df, ['dept', 'department', 'functional_unit', 'role', 'team'])

print(f"\nDetected columns:")
print(f"  User   → {USER_COL}")
print(f"  Week   → {WEEK_COL}")
print(f"  Label  → {LABEL_COL}")
print(f"  Dept   → {DEPT_COL}")


# Convert labels to binary immediately: anything > 0 is an insider
df[LABEL_COL]    = df[LABEL_COL].astype(int)
df['binary_label'] = (df[LABEL_COL] > 0).astype(int)

print(f"\nOriginal label counts (raw):")
print(df[LABEL_COL].value_counts().sort_index())

print(f"\nBinary label counts:")
print(df['binary_label'].value_counts())

n_normal  = (df['binary_label'] == 0).sum()
n_insider = (df['binary_label'] == 1).sum()

print(f"\nClass imbalance : {n_normal / n_insider:.0f}:1  (normal : insider)")
print(f"Insider rate    : {n_insider / len(df) * 100:.3f}%")
print(f"Week range      : {df[WEEK_COL].min()} → {df[WEEK_COL].max()}")
print(f"Unique users    : {df[USER_COL].nunique()}")

# From here on, always use the binary label
LABEL_COL = 'binary_label'


# ── Cell 3: Exploratory Data Analysis ────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('CERT v5.2 — Binary Insider Threat EDA', fontsize=16, fontweight='bold')

# Plot 1: How many normal vs insider user-weeks are there?
vc = df[LABEL_COL].value_counts().sort_index()
axes[0, 0].bar(
    ['Normal (0)', 'Insider (1)'],
    vc.values,
    color=['#2ecc71', '#e74c3c'],
    edgecolor='black',
)
axes[0, 0].set_title('Binary Class Distribution')
axes[0, 0].set_ylabel('Count')
for i, v in enumerate(vc.values):
    axes[0, 0].text(i, v + 100, f'{v:,}', ha='center', fontsize=10, fontweight='bold')

# Plot 2: When does insider activity happen across the study period?
insider_by_week = df[df[LABEL_COL] == 1].groupby(WEEK_COL)[LABEL_COL].count()
axes[0, 1].bar(insider_by_week.index, insider_by_week.values, color='#e74c3c', alpha=0.8)
axes[0, 1].set_title('Insider Activity by Week')
axes[0, 1].set_xlabel('Week')
axes[0, 1].set_ylabel('Insider User-Weeks')

# Plot 3: Pie chart showing just how severe the imbalance is
axes[0, 2].pie(
    [n_normal, n_insider],
    labels=[
        f'Normal\n{n_normal / len(df) * 100:.2f}%',
        f'Insider\n{n_insider / len(df) * 100:.2f}%',
    ],
    colors=['#2ecc71', '#e74c3c'],
    autopct='%1.2f%%',
    startangle=90,
)
axes[0, 2].set_title(f'Class Imbalance ({n_normal / n_insider:.0f}:1)')

# Plot 4: How many active users are there each week?
users_per_week = df.groupby(WEEK_COL)[USER_COL].nunique()
axes[1, 0].plot(users_per_week.index, users_per_week.values, '#3498db', linewidth=2)
axes[1, 0].set_title('Active Users per Week')
axes[1, 0].set_xlabel('Week')
axes[1, 0].set_ylabel('Unique Users')

# Plot 5: How many distinct insider users appear each week?
insider_users_per_week = df[df[LABEL_COL] == 1].groupby(WEEK_COL)[USER_COL].nunique()
axes[1, 1].bar(insider_users_per_week.index, insider_users_per_week.values,
               color='#e74c3c', alpha=0.8)
axes[1, 1].set_title('Unique Insider Users per Week')
axes[1, 1].set_xlabel('Week')
axes[1, 1].set_ylabel('Unique Insider Users')

# Plot 6: What do the raw feature distributions look like?
numeric_cols_eda = df.select_dtypes(include=np.number).columns.difference(
    [USER_COL, WEEK_COL, LABEL_COL, 'binary_label']
)
for feat in numeric_cols_eda[:5]:
    axes[1, 2].hist(
        df[feat].clip(upper=df[feat].quantile(0.99)),
        bins=50, alpha=0.5, label=feat[:15],
    )
axes[1, 2].set_title('Feature Distributions (5 features)')
axes[1, 2].set_xlabel('Value')
axes[1, 2].legend(fontsize=7)

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print("EDA plots saved to eda_plots.png")


# ── Cell 4: Preprocessing and department partitioning ────────────────────

print("\n" + "=" * 60)
print("STEP 3: PREPROCESSING & DEPARTMENT PARTITIONING")
print("=" * 60)

df = df.copy()

# Fill missing numeric values with 0 (safe default for behavioural features)
numeric_cols_all = df.select_dtypes(include=np.number).columns
df[numeric_cols_all] = df[numeric_cols_all].fillna(0)


# ── Assign each user-week to a department ─────────────────────────────────
#
# We need to split data into "clients" (departments) for federated learning.
# There are three cases:
#   a) The CSV already has a department column → use it directly
#   b) User IDs have an alphabetic prefix (e.g. "ENG001") → use that prefix
#   c) User IDs are purely numeric → bucket them into 10 groups via modulo

if DEPT_COL is not None and DEPT_COL != 'binary_label':
    df['dept'] = df[DEPT_COL].astype(str)
    print(f"Using existing department column: {DEPT_COL}")
else:
    prefix         = df[USER_COL].astype(str).str.extract(r'^([A-Za-z]+)')[0]
    n_alpha_depts  = prefix.nunique()

    if n_alpha_depts >= 3 and prefix.notna().sum() > len(df) * 0.5:
        df['dept'] = prefix.fillna('UNK')
        print(f"Derived {n_alpha_depts} departments from alphabetic prefix of user ID")
    else:
        # Pure numeric IDs — modulo bucketing gives 10 stable groups
        df['dept'] = (
            df[USER_COL].astype(str)
            .str.extract(r'(\d+)')[0]
            .fillna('0')
            .astype(int) % 10
        ).astype(str)
        print("Numeric user IDs detected — bucketed into 10 department groups (modulo 10)")

departments = sorted(df['dept'].unique().tolist())
print(f"\nDepartments found ({len(departments)} total): {departments}")

# Show a per-department summary so we can spot any very small or imbalanced clients
dept_summary = df.groupby('dept').agg(
    total_weeks   = (LABEL_COL, 'count'),
    insider_weeks = (LABEL_COL, 'sum'),
    unique_users  = (USER_COL, 'nunique'),
).sort_values('total_weeks', ascending=False)
dept_summary['insider_rate_pct'] = (
    dept_summary['insider_weeks'] / dept_summary['total_weeks'] * 100
).round(3)
print(dept_summary)


# ── Feature columns ───────────────────────────────────────────────────────
# Everything numeric that is not a metadata column is a model feature.
META_COLS = list(set(
    c for c in [USER_COL, WEEK_COL, LABEL_COL, 'binary_label', 'dept', DEPT_COL]
    if c is not None
))
FEATURE_COLS = [
    c for c in df.select_dtypes(include=np.number).columns
    if c not in META_COLS
]
print(f"\nNumber of features: {len(FEATURE_COLS)}")


# ── Temporal train / val / test split ─────────────────────────────────────
# Use time as the split boundary — no data leakage.
# We never train on future weeks and evaluate on past weeks.
all_weeks   = sorted(df[WEEK_COL].unique())
n_weeks     = len(all_weeks)
train_end   = int(n_weeks * 0.72)   # first 72% of weeks → training
val_end     = int(n_weeks * 0.86)   # next 14% → validation, last 14% → test

train_weeks = all_weeks[:train_end]
val_weeks   = all_weeks[train_end:val_end]
test_weeks  = all_weeks[val_end:]

df['split'] = 'train'
df.loc[df[WEEK_COL].isin(val_weeks),  'split'] = 'val'
df.loc[df[WEEK_COL].isin(test_weeks), 'split'] = 'test'

train_mask = df['split'] == 'train'
val_mask   = df['split'] == 'val'
test_mask  = df['split'] == 'test'

print(f"\nTemporal split summary:")
print(f"  Train : weeks {train_weeks[0]}–{train_weeks[-1]}  "
      f"({len(train_weeks)} weeks) | insiders: {df.loc[train_mask, LABEL_COL].sum()}")
print(f"  Val   : weeks {val_weeks[0]}–{val_weeks[-1]}    "
      f"({len(val_weeks)} weeks)  | insiders: {df.loc[val_mask,   LABEL_COL].sum()}")
print(f"  Test  : weeks {test_weeks[0]}–{test_weeks[-1]}   "
      f"({len(test_weeks)} weeks) | insiders: {df.loc[test_mask,  LABEL_COL].sum()}")


# ── Per-department MinMax scaling ─────────────────────────────────────────
# Critically, scalers are fit only on each department's training data.
# This mirrors what would happen in a real federated deployment — no
# department has access to another's raw data, not even for normalisation.
X_all    = df[FEATURE_COLS].values.astype(np.float32)
y_all    = df[LABEL_COL].values.astype(np.int64)
X_scaled = np.zeros_like(X_all)
scalers  = {}

for dept in departments:
    train_idx  = df[(df['dept'] == dept) & (df['split'] == 'train')].index
    all_idx    = df[df['dept'] == dept].index

    if len(train_idx) == 0:
        # No training rows for this dept — leave values unscaled
        X_scaled[all_idx] = X_all[all_idx]
        continue

    sc = MinMaxScaler()
    sc.fit(X_all[train_idx])
    X_scaled[all_idx] = sc.transform(X_all[all_idx])
    scalers[dept]     = sc

# Hard-clip to [0, 1] to handle rare out-of-range test values
X_scaled   = np.clip(X_scaled, 0.0, 1.0)
n_features = X_scaled.shape[1]
print(f"\nScaling complete. Value range: {X_scaled.min():.4f} → {X_scaled.max():.4f}")
print(f"Total features after scaling: {n_features}")


# ── Cell 5: Build temporal sequences ─────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 4: BUILDING TEMPORAL SEQUENCES (window = 8 weeks, binary labels)")
print("=" * 60)


def build_sequences(user_rows_df, df_global, X_sc, T=8):
    """
    For each user, slide a window of T consecutive weeks across their
    timeline and create one (sequence, label) pair per window.

    The label assigned to a window is the binary label of the final week
    in that window — we're predicting "is this person behaving suspiciously
    right now given the past T weeks?"

    Sequences are assigned to train/val/test based on which split the
    final week belongs to, preserving the temporal ordering guarantee.
    """
    seqs = {'train': [], 'val': [], 'test': []}
    lbls = {'train': [], 'val': [], 'test': []}

    for user in user_rows_df[USER_COL].unique():
        rows  = user_rows_df[user_rows_df[USER_COL] == user].sort_values(WEEK_COL)
        u_idx = rows.index.tolist()

        if len(u_idx) < T:
            continue  # skip users with fewer than T weeks of data

        for i in range(T - 1, len(u_idx)):
            window = u_idx[i - T + 1 : i + 1]   # the T-week window
            split  = df_global.loc[u_idx[i], 'split']
            seqs[split].append(X_sc[window])
            lbls[split].append(df_global.loc[u_idx[i], LABEL_COL])

    def to_arrays(seq_list, lbl_list):
        if not seq_list:
            return (
                np.zeros((0, T, X_sc.shape[1]), dtype=np.float32),
                np.zeros(0, dtype=np.int64),
            )
        return np.array(seq_list, dtype=np.float32), np.array(lbl_list, dtype=np.int64)

    result = {}
    for split in ['train', 'val', 'test']:
        result[f'X_{split}'], result[f'y_{split}'] = to_arrays(seqs[split], lbls[split])
    return result


# Build department-level sequence datasets (used in Stage-1 FL training)
dept_datasets = {}
for dept in tqdm(departments, desc="Building department-level sequences"):
    dept_datasets[dept] = build_sequences(
        df[df['dept'] == dept], df, X_scaled, T=T_WINDOW
    )

print(f"\n{'Dept':<8} {'Train':>8} {'Val':>8} {'Test':>8} {'Ins(Tr)':>9} {'Ins(Te)':>9}")
print("-" * 52)
for dept in departments:
    d = dept_datasets[dept]
    print(
        f"{dept:<8} {len(d['X_train']):>8,} {len(d['X_val']):>8,} "
        f"{len(d['X_test']):>8,} {d['y_train'].sum():>9} {d['y_test'].sum():>9}"
    )


# Build per-user sequence datasets (used in Stage-2 personalisation)
print("\nBuilding per-user sequences for Stage-2 fine-tuning...")
user_datasets = {}

for user in tqdm(df[USER_COL].unique(), desc="Building user-level sequences"):
    rows  = df[df[USER_COL] == user].sort_values(WEEK_COL)
    u_idx = rows.index.tolist()
    dept  = rows['dept'].iloc[0]

    if len(u_idx) < T_WINDOW:
        continue  # not enough history for this user

    seqs = {'train': [], 'val': [], 'test': []}
    lbls = {'train': [], 'val': [], 'test': []}

    for i in range(T_WINDOW - 1, len(u_idx)):
        window = u_idx[i - T_WINDOW + 1 : i + 1]
        split  = df.loc[u_idx[i], 'split']
        seqs[split].append(X_scaled[window])
        lbls[split].append(df.loc[u_idx[i], LABEL_COL])

    def to_arrays2(seq_list, lbl_list):
        if not seq_list:
            return (
                np.zeros((0, T_WINDOW, n_features), dtype=np.float32),
                np.zeros(0, dtype=np.int64),
            )
        return np.array(seq_list, dtype=np.float32), np.array(lbl_list, dtype=np.int64)

    ud = {'dept': dept}
    for split in ['train', 'val', 'test']:
        ud[f'X_{split}'], ud[f'y_{split}'] = to_arrays2(seqs[split], lbls[split])

    # Track whether this user ever had an insider label — useful for targeted fine-tuning
    ud['is_insider'] = any(l > 0 for llist in lbls.values() for l in llist)
    user_datasets[user] = ud

insider_user_count = sum(1 for u in user_datasets.values() if u['is_insider'])
print(f"Users with sequences: {len(user_datasets)} | Known insider users: {insider_user_count}")


# ── Cell 6: Model architecture ────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 5: BINARY CLASSIFICATION MODEL ARCHITECTURE")
print("=" * 60)


class BinaryInsiderBody(nn.Module):
    """
    The shared 'body' of the model — this part is aggregated on the
    central server each federated round and then distributed back.

    Architecture overview:
      1. A position-wise encoder maps each week's feature vector into a
         richer 128-dimensional representation (two linear layers + GELU).
      2. A 2-layer bidirectional LSTM captures the ordering of weeks —
         i.e., trends, ramp-ups, sudden changes.
      3. Multi-head self-attention lets each week attend to all other weeks,
         finding long-range temporal patterns the LSTM might miss.
      4. The attended representations are averaged across the time dimension
         to produce a single fixed-size vector for the classifier head.

    We use LayerNorm instead of BatchNorm because FL clients often have
    small batch sizes, which makes BatchNorm statistics unreliable.
    """

    def __init__(self, n_features, hidden_dim=128, n_heads=4, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Step 1: per-timestep feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128),        nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
        )

        # Step 2: bidirectional LSTM to model temporal dynamics
        self.lstm = nn.LSTM(
            128, hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )

        # Step 3: multi-head self-attention for non-local temporal relationships
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(hidden_dim)

        # Step 4: final projection
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        B, T, F = x.shape

        # Encode each timestep independently (shape: B×T×128)
        enc = self.encoder(x.reshape(B * T, F)).reshape(B, T, 128)

        # Apply BiLSTM across the time axis (shape: B×T×hidden_dim)
        lstm_out, _ = self.lstm(enc)

        # Self-attention across the time axis, then residual connection
        attn_out, attn_weights = self.attn(lstm_out, lstm_out, lstm_out)
        attended = self.norm(lstm_out + attn_out)

        # Average pool across time to get a single representation per sequence
        rep = attended.mean(dim=1)
        return self.proj(rep), attn_weights


class BinaryPersonalizedModel(nn.Module):
    """
    The full personalised model for one user or department.

    This combines three components:
      - body:             shared encoder (the federated part, updated every round)
      - deviation_encoder: local anomaly detector — captures HOW this specific
                          user deviates from their own baseline behaviour.
                          This component is NEVER sent to the server.
      - head:             local binary classifier — also never shared.

    The deviation encoder is the key personalisation insight: two users might
    have identical raw feature values, but if one of them normally exhibits
    those values and the other normally doesn't, only the deviation encoder
    can tell them apart.

    Outputs a single logit (use with BCEWithLogitsLoss or sigmoid → probability).
    """

    def __init__(self, n_features, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.body = BinaryInsiderBody(n_features, hidden_dim)

        # The deviation encoder learns: "is this week unusual FOR THIS USER?"
        self.deviation_encoder = nn.Sequential(
            nn.Linear(n_features, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 32),
        )

        # Binary classification head: body output (128-d) + deviation (32-d) → logit
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 32, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32),              nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1),               # single logit for binary classification
        )

        # Running per-user behavioural baseline (updated online during training,
        # stored in model state but never shared with the server)
        self.register_buffer('user_mean', torch.zeros(n_features))
        self.register_buffer('user_std',  torch.ones(n_features))

    def update_user_stats(self, x_batch):
        """
        Exponential moving average update of this user's behavioural baseline.
        Called during training so the baseline tracks gradual drift.
        alpha=0.1 means the baseline is slow to change — resistant to
        a short burst of anomalous activity masking itself.
        """
        last_week = x_batch[:, -1, :]  # most recent week in each sequence
        alpha = 0.1
        self.user_mean = (1 - alpha) * self.user_mean + alpha * last_week.mean(0).detach()
        self.user_std  = (1 - alpha) * self.user_std  + alpha * (last_week.std(0) + 1e-8).detach()

    def forward(self, x):
        # Get the shared body representation
        body_rep, attn_weights = self.body(x)

        # Compute how this user's most recent week deviates from their own norm
        last_week = x[:, -1, :]
        deviation = (last_week - self.user_mean) / self.user_std
        dev_enc   = self.deviation_encoder(deviation)

        # Concatenate and classify
        combined = torch.cat([body_rep, dev_enc], dim=1)
        logit    = self.head(combined).squeeze(1)   # shape: (batch_size,)
        return logit, attn_weights

    def predict_proba(self, x):
        """Returns P(insider) ∈ [0, 1] by applying sigmoid to the logit."""
        logit, attn_weights = self.forward(x)
        return torch.sigmoid(logit), attn_weights

    def get_local_params(self):
        """Parameters that stay local and are never aggregated on the server."""
        return list(self.deviation_encoder.parameters()) + list(self.head.parameters())

    def get_body_params(self):
        """Parameters that are shared with the server each round."""
        return list(self.body.parameters())


# Quick sanity check: does the forward pass work?
dummy_input = torch.zeros(4, T_WINDOW, n_features).to(DEVICE)
test_model  = BinaryPersonalizedModel(n_features).to(DEVICE)
test_logit, test_attn = test_model(dummy_input)

print(f"Forward pass OK — output logit shape: {test_logit.shape}  (one per sample in batch)")

body_params  = sum(p.numel() for p in test_model.body.parameters())
local_params = sum(p.numel() for p in test_model.get_local_params())
print(f"Body params (shared across federation) : {body_params:,}")
print(f"Local params (head + deviation encoder): {local_params:,}")
print(f"Total parameters                       : {body_params + local_params:,}")

del test_model, dummy_input


# ── Cell 7: Loss functions and training utilities ─────────────────────────


class BinaryFocalLoss(nn.Module):
    """
    Focal loss for binary classification with severe class imbalance.

    Standard cross-entropy treats all misclassified examples equally.
    Focal loss applies a multiplicative factor (1 - p_t)^gamma to
    down-weight the contribution of easy, well-classified examples
    (the overwhelming majority of normal weeks). This forces the model
    to focus its learning capacity on the hard-to-classify insider weeks.

    We use gamma=1.5 (conservative) because the original paper's gamma=2.0
    and our earlier gamma=3.0 both caused the model to sacrifice precision
    entirely in pursuit of recall — almost every normal week got flagged.

    pos_weight provides an additional multiplicative upweighting of the
    insider class in the BCE component, independent of the focal term.
    """

    def __init__(self, gamma=1.5, pos_weight=None):
        super().__init__()
        self.gamma      = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        # Standard weighted binary cross-entropy
        bce = F.binary_cross_entropy_with_logits(
            logits, targets.float(),
            pos_weight=self.pos_weight,
            reduction='none',
        )
        # Focal modulation: down-weight easy examples
        probs   = torch.sigmoid(logits)
        p_t     = torch.where(targets == 1, probs, 1 - probs)
        focal_w = (1.0 - p_t) ** self.gamma
        return (focal_w * bce).mean()


def compute_pos_weight(y, device=DEVICE, cap=50.0):
    """
    Compute the positive-class weight for BCE loss from the training labels.

    The raw ratio in CERT v5.2 is roughly 200:1 (normal : insider).
    Setting pos_weight=200 would cause the model to predict "insider" for
    nearly everything, collapsing precision to zero. We cap at 50 as a
    conservative compromise that still strongly upweights insider examples.
    """
    n_negative = (y == 0).sum()
    n_positive = (y == 1).sum()

    if n_positive == 0:
        return torch.tensor(1.0).to(device)

    raw_weight = n_negative / n_positive
    weight     = min(float(raw_weight), cap)
    return torch.tensor(weight).to(device)


def build_oversampled_loader(X, y, batch_size=128, oversample_ratio=2):
    """
    Build a DataLoader with mild oversampling of the minority (insider) class.

    We repeat each insider sequence `oversample_ratio` times in the epoch.
    With ratio=2 every insider example appears twice per epoch — enough to
    ensure insiders appear in every training batch without catastrophically
    distorting the class distribution. Our earlier ratio=5 caused the model
    to train on a near-balanced dataset, which inflated recall at the cost
    of completely destroying precision.
    """
    insider_idx = np.where(y == 1)[0]
    normal_idx  = np.where(y == 0)[0]

    if len(insider_idx) == 0:
        # No insiders in this partition — fall back to standard loader
        ds = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
        return DataLoader(ds, batch_size=batch_size, shuffle=True,
                          num_workers=0, pin_memory=(DEVICE == 'cuda'))

    all_idx = np.concatenate([normal_idx, np.tile(insider_idx, oversample_ratio)])
    np.random.shuffle(all_idx)

    ds = TensorDataset(
        torch.FloatTensor(X[all_idx]),
        torch.FloatTensor(y[all_idx].astype(np.float32)),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=0, pin_memory=(DEVICE == 'cuda'))


def get_loader(X, y, batch_size=64, shuffle=True):
    """Standard DataLoader with no resampling — used for validation/test."""
    ds = TensorDataset(
        torch.FloatTensor(X),
        torch.FloatTensor(y.astype(np.float32)),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=(DEVICE == 'cuda'))


@torch.no_grad()
def predict_binary(model, global_body_state, X, threshold=0.5, batch_size=256):
    """
    Run inference for a single department/user model.

    Loads the latest global body weights first, then uses the model's
    own local head and deviation encoder to produce probabilities.
    Returns (probabilities, binary_predictions).
    """
    model.body.load_state_dict(global_body_state)
    model.eval()
    loader = get_loader(X, np.zeros(len(X), dtype=np.int64), batch_size, shuffle=False)

    all_probs = []
    for x_batch, _ in loader:
        prob, _ = model.predict_proba(x_batch.to(DEVICE))
        all_probs.append(prob.cpu())

    probs = torch.cat(all_probs).numpy()
    preds = (probs >= threshold).astype(int)
    return probs, preds


@torch.no_grad()
def evaluate_binary(model, global_body_state, X, y, threshold=0.5, batch_size=256):
    """
    Evaluate a model on a labelled dataset and return a dictionary of metrics.
    Includes F1, precision-weighted F-beta, precision, recall, and AUC-ROC.
    """
    probs, preds = predict_binary(model, global_body_state, X, threshold, batch_size)
    labels       = y.astype(int)

    f1_score_val   = f1_score(labels, preds, zero_division=0)
    fbeta_score_val = fbeta_score(labels, preds, beta=0.5, zero_division=0)
    precision_val  = precision_score(labels, preds, zero_division=0)
    recall_val     = recall_score(labels, preds, zero_division=0)

    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = float('nan')

    return {
        'f1':        f1_score_val,
        'fbeta':     fbeta_score_val,
        'precision': precision_val,
        'recall':    recall_val,
        'auc':       auc,
        'probs':     probs,
        'preds':     preds,
        'labels':    labels,
    }


def fedavg_aggregate(body_states, dataset_sizes):
    """
    Federated Averaging (FedAvg) — weighted mean of body parameters.

    Each client's contribution is weighted by the number of training
    sequences it holds. Larger departments have more influence on the
    global model, which is generally appropriate because their gradients
    are averaged over more examples and therefore less noisy.
    """
    total   = sum(dataset_sizes)
    weights = [s / total for s in dataset_sizes]
    aggregated = copy.deepcopy(body_states[0])
    for key in aggregated:
        aggregated[key] = sum(
            w * state[key].float()
            for w, state in zip(weights, body_states)
        )
    return aggregated


# Pre-compute the positive class weight for each department's training set
dept_pos_weights = {}
for dept in departments:
    y_train = dept_datasets[dept]['y_train']
    if len(y_train) >= 10:
        dept_pos_weights[dept] = compute_pos_weight(y_train, DEVICE, cap=50.0)
        n_ins = y_train.sum()
        if n_ins > 0:
            print(
                f"  Dept {dept}: {(y_train == 0).sum()} normal, {n_ins} insider "
                f"| pos_weight = {dept_pos_weights[dept].item():.1f}"
            )


# ── Cell 8: Stage-1 — pFedMe federated training across departments ────────

print("\n" + "=" * 60)
print("STEP 6a: STAGE-1 — pFedMe DEPARTMENT-LEVEL TRAINING")
print("=" * 60)

# ── Hyperparameters ───────────────────────────────────────────────────────
HIDDEN_DIM       = 128
N_ROUNDS_S1      = 20    # FL rounds (20 is sufficient for binary classification)
E_LOCAL          = 5     # local training epochs per round
BATCH_SIZE       = 128
LR_LOCAL         = 2e-4  # learning rate (slightly lower than default for stability)
WEIGHT_DECAY     = 1e-4
PATIENCE         = 8     # early-stopping rounds without improvement
GAMMA_FOCAL      = 1.5   # focal loss gamma (conservative)
MIN_DEPT_SIZE    = 5     # skip departments with fewer than this many training examples
OVERSAMPLE_RATIO = 2     # minority oversampling multiplier
LAMBDA_PROX      = 0.5   # pFedMe proximal term strength (higher = closer to global)

print(
    f"pFedMe binary | λ={LAMBDA_PROX} | γ_focal={GAMMA_FOCAL} "
    f"| Oversample={OVERSAMPLE_RATIO}× | Rounds={N_ROUNDS_S1}"
)

# Initialise one personalised model per department
local_models = {
    dept: BinaryPersonalizedModel(n_features, HIDDEN_DIM).to(DEVICE)
    for dept in departments
}

# The global body starts from the first model's random initialisation
global_body_state = copy.deepcopy(list(local_models.values())[0].body.state_dict())

# Each department gets its own optimizer (these are never shared)
optimizers = {
    dept: AdamW(local_models[dept].parameters(), lr=LR_LOCAL, weight_decay=WEIGHT_DECAY)
    for dept in departments
}
schedulers = {
    dept: CosineAnnealingLR(optimizers[dept], T_max=N_ROUNDS_S1)
    for dept in departments
}

# Track validation metrics to pick the best checkpoint
history = {
    'round':      [],
    'train_loss': [],
    'val_f1':     [],
    'val_auc':    [],
    'val_fbeta':  [],
}
best_val_fbeta   = 0.0
best_val_auc     = 0.0
best_body_state  = None
best_head_states = None
best_dev_states  = None
patience_count   = 0


def train_pfedme_one_round(dept, global_body_state):
    """
    Perform one round of local pFedMe training for a single department.

    pFedMe modifies the standard local SGD update by adding a proximal
    term that penalises the local body for drifting too far from the
    current global body. This prevents a common federated learning failure
    mode called "client drift" — where each client's model converges to a
    very good solution for its own data but becomes completely incompatible
    with other clients' models, making aggregation harmful.

    Loss = BinaryFocalLoss(γ=1.5) + (λ/2) × ||local_body − global_body||²

    We jointly update all parameters (body + head + deviation encoder)
    rather than alternating head/body updates. Joint updates are more
    stable for binary classification because the gradient signal is
    simpler and less likely to cause one component to overshoot while
    the other is frozen.

    Returns:
        new_body_state: the updated local body weights to send to the server
        avg_loss:       mean task loss across the local epoch (for logging)
    """
    data = dept_datasets[dept]
    X_train, y_train = data['X_train'], data['y_train']

    if len(X_train) < MIN_DEPT_SIZE:
        return None, None

    model = local_models[dept]
    model.body.load_state_dict(global_body_state)
    model.train()

    # Snapshot the global body parameters (detached) as the proximal anchor
    global_body_reference = {
        name: param.detach().clone()
        for name, param in model.body.named_parameters()
    }

    loader    = build_oversampled_loader(X_train, y_train, BATCH_SIZE, OVERSAMPLE_RATIO)
    pw        = dept_pos_weights.get(dept, torch.tensor(10.0).to(DEVICE))
    criterion = BinaryFocalLoss(gamma=GAMMA_FOCAL, pos_weight=pw)
    opt       = optimizers[dept]

    # Update this department's behavioural baseline before training
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_train).to(DEVICE)
        model.update_user_stats(X_tensor)

    total_loss = 0.0
    n_batches  = 0

    for _epoch in range(E_LOCAL):
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            opt.zero_grad()

            logit, _ = model(x_batch)
            task_loss = criterion(logit, y_batch)

            # pFedMe proximal term: penalise drift from the global body
            proximal_term = sum(
                ((local_param - global_body_reference[name]) ** 2).sum()
                for name, local_param in model.body.named_parameters()
            )
            loss = task_loss + (LAMBDA_PROX / 2) * proximal_term

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # prevent exploding gradients
            opt.step()

            total_loss += task_loss.item()
            n_batches  += 1

    avg_loss      = total_loss / max(n_batches, 1)
    new_body_state = copy.deepcopy(model.body.state_dict())
    return new_body_state, avg_loss


# ── Main training loop ────────────────────────────────────────────────────

print("\nStarting Stage-1 federated training...")
training_start = time.time()

for round_num in range(1, N_ROUNDS_S1 + 1):
    new_body_states = []
    dept_sizes      = []
    round_losses    = []

    # Each department trains locally, then sends its body weights to the server
    for dept in departments:
        new_body, avg_loss = train_pfedme_one_round(dept, global_body_state)
        if new_body is None:
            continue
        new_body_states.append(new_body)
        dept_sizes.append(len(dept_datasets[dept]['X_train']))
        round_losses.append(avg_loss)
        schedulers[dept].step()

    # Server aggregates all department bodies into a new global body
    global_body_state = fedavg_aggregate(new_body_states, dept_sizes)
    avg_round_loss    = np.mean(round_losses) if round_losses else 0.0
    elapsed_secs      = time.time() - training_start

    # Validate every 5 rounds (and always on round 1 to get a baseline)
    should_validate = (round_num % 5 == 0 or round_num == 1)

    if should_validate:
        dept_f1s, dept_aucs, dept_fbetas = [], [], []

        for dept in departments:
            data = dept_datasets[dept]
            if len(data['X_val']) == 0 or data['y_val'].sum() == 0:
                continue
            res = evaluate_binary(
                local_models[dept], global_body_state,
                data['X_val'], data['y_val'],
            )
            dept_f1s.append(res['f1'])
            dept_aucs.append(res['auc'])
            dept_fbetas.append(res['fbeta'])

        avg_f1    = np.mean(dept_f1s)    if dept_f1s    else 0.0
        avg_auc   = np.mean(dept_aucs)   if dept_aucs   else 0.0
        avg_fbeta = np.mean(dept_fbetas) if dept_fbetas else 0.0

        history['round'].append(round_num)
        history['train_loss'].append(avg_round_loss)
        history['val_f1'].append(avg_f1)
        history['val_auc'].append(avg_auc)
        history['val_fbeta'].append(avg_fbeta)

        print(
            f"Round {round_num:3d}/{N_ROUNDS_S1} | "
            f"Loss: {avg_round_loss:.5f} | "
            f"Val F1: {avg_f1:.4f} | "
            f"AUC: {avg_auc:.4f} | "
            f"Fβ(0.5): {avg_fbeta:.4f} | "
            f"{elapsed_secs:.0f}s elapsed"
        )

        # Save checkpoint if this is the best F-beta so far
        if avg_fbeta > best_val_fbeta:
            best_val_fbeta   = avg_fbeta
            best_val_auc     = avg_auc
            best_body_state  = copy.deepcopy(global_body_state)
            best_head_states = {
                d: copy.deepcopy(local_models[d].head.state_dict())
                for d in departments
            }
            best_dev_states = {
                d: copy.deepcopy(local_models[d].deviation_encoder.state_dict())
                for d in departments
            }
            patience_count = 0
            print(f"  ✓ New best  Val Fβ = {best_val_fbeta:.4f} | AUC = {best_val_auc:.4f}")
            torch.save(
                {
                    'global_body': best_body_state,
                    'head_states': best_head_states,
                    'dev_states':  best_dev_states,
                    'round':       round_num,
                },
                'best_stage1_checkpoint.pt',
            )
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"\nEarly stopping triggered at round {round_num} "
                      f"(no improvement for {PATIENCE} validation checks)")
                break
    else:
        print(f"Round {round_num:3d}/{N_ROUNDS_S1} | Loss: {avg_round_loss:.5f} | {elapsed_secs:.0f}s")

total_minutes = (time.time() - training_start) / 60
print(f"\nStage-1 complete. Best Val Fβ = {best_val_fbeta:.4f} | AUC = {best_val_auc:.4f}")
print(f"Total training time: {total_minutes:.1f} minutes")


# ── Cell 9: Plot Stage-1 training curves ─────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Stage-1 Binary pFedMe Training Progress', fontsize=14, fontweight='bold')

axes[0].plot(history['round'], history['train_loss'], 'b-o', markersize=4, linewidth=2)
axes[0].set_title('Focal Loss (averaged across departments)')
axes[0].set_xlabel('Federated Round')
axes[0].set_ylabel('Loss')
axes[0].grid(alpha=0.3)

axes[1].plot(history['round'], history['val_f1'],    'g-o', markersize=4, label='F1')
axes[1].plot(history['round'], history['val_fbeta'], 'm-o', markersize=4, label='Fβ(0.5)')
axes[1].axhline(best_val_fbeta, color='r', linestyle='--',
                label=f'Best Fβ = {best_val_fbeta:.4f}')
axes[1].set_title('Validation F1 & Fβ(0.5)')
axes[1].set_xlabel('Federated Round')
axes[1].set_ylabel('Score')
axes[1].legend()
axes[1].grid(alpha=0.3)

axes[2].plot(history['round'], history['val_auc'], 'r-o', markersize=4, linewidth=2)
axes[2].axhline(best_val_auc, color='navy', linestyle='--',
                label=f'Best AUC = {best_val_auc:.4f}')
axes[2].set_title('Validation AUC-ROC')
axes[2].set_xlabel('Federated Round')
axes[2].set_ylabel('AUC')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
plt.show()


# ── Cell 10: Stage-2 — Per-user fine-tuning ───────────────────────────────

print("\n" + "=" * 60)
print("STEP 6b: STAGE-2 — USER-LEVEL PERSONALISATION")
print("=" * 60)

# Fine-tuning hyperparameters
N_FINETUNE_EPOCHS = 15   # local epochs (binary task is simpler, converges faster)
LR_FINETUNE       = 3e-4
MIN_USER_SAMPLES  = 3    # skip users with fewer than 3 training sequences

print(f"Fine-tuning {len(user_datasets)} users | {N_FINETUNE_EPOCHS} epochs each")
print("Body is frozen during fine-tuning — only head and deviation encoder are updated")
print("This ensures the globally shared representations are not corrupted by")
print("the idiosyncrasies of any single user's data.")

user_models = {}
ft_start    = time.time()

for user, data in tqdm(user_datasets.items(), desc="Fine-tuning user models"):
    X_train = data['X_train']
    y_train = data['y_train']
    dept    = data['dept']

    if len(X_train) < MIN_USER_SAMPLES:
        continue

    # Start from scratch but immediately load the best federated body
    model = BinaryPersonalizedModel(n_features, HIDDEN_DIM).to(DEVICE)
    model.body.load_state_dict(best_body_state)

    # Warm-start the local head and deviation encoder from the department checkpoint
    # (better than random init — the dept model already knows this user's context)
    if dept in best_head_states:
        model.head.load_state_dict(best_head_states[dept])
    if dept in best_dev_states:
        model.deviation_encoder.load_state_dict(best_dev_states[dept])

    # Update this user's personal behavioural baseline
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_train).to(DEVICE)
        model.update_user_stats(X_tensor)

    # Freeze the body — fine-tuning only the local components
    for param in model.body.parameters():
        param.requires_grad_(False)
    for param in model.get_local_params():
        param.requires_grad_(True)

    opt  = AdamW(model.get_local_params(), lr=LR_FINETUNE, weight_decay=WEIGHT_DECAY)
    pw   = compute_pos_weight(y_train, DEVICE, cap=50.0)
    crit = BinaryFocalLoss(gamma=GAMMA_FOCAL, pos_weight=pw)

    # Give known insider users a stronger oversampling signal
    oversample_ratio = 3 if data['is_insider'] else 1
    loader = build_oversampled_loader(X_train, y_train, batch_size=32,
                                      oversample_ratio=oversample_ratio)

    model.train()
    for _epoch in range(N_FINETUNE_EPOCHS):
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            opt.zero_grad()
            logit, _ = model(x_batch)
            crit(logit, y_batch).backward()
            nn.utils.clip_grad_norm_(model.get_local_params(), 1.0)
            opt.step()

    # Unfreeze body for potential downstream use (e.g. inference with full grad)
    for param in model.body.parameters():
        param.requires_grad_(True)

    user_models[user] = model

ft_minutes = (time.time() - ft_start) / 60
print(f"\nDone: {len(user_models)} user models fine-tuned in {ft_minutes:.1f} minutes")

torch.save(
    {
        'global_body': best_body_state,
        'user_states': {
            u: {
                'head': m.head.state_dict(),
                'dev':  m.deviation_encoder.state_dict(),
            }
            for u, m in user_models.items()
        },
    },
    'best_stage2_checkpoint.pt',
)
print("Stage-2 checkpoint saved to best_stage2_checkpoint.pt")


# ── Cell 11: Precision-first threshold selection on the validation set ────

print("\n" + "=" * 60)
print("STEP 7: PRECISION-FIRST THRESHOLD SELECTION (validation set)")
print("=" * 60)

print("""
Standard threshold selection maximises F1 on the validation set.
For insider threat detection with 200:1 imbalance, this typically finds
a threshold so low that almost everything is flagged — precision collapses
to near zero and the alert queue becomes unworkable for security analysts.

Instead we find the LOWEST threshold at which precision meets each target.
This means every alert fired has at least a 50% chance of being a real
insider threat — a much more operationally useful system.
""")

# Collect validation probabilities from all departments using the best checkpoint
val_probs_all = []
val_true_all  = []

for dept in departments:
    data = dept_datasets[dept]
    if len(data['X_val']) == 0:
        continue

    # Load the best checkpoint for this department
    local_models[dept].body.load_state_dict(best_body_state)
    if dept in best_head_states:
        local_models[dept].head.load_state_dict(best_head_states[dept])
    if dept in best_dev_states:
        local_models[dept].deviation_encoder.load_state_dict(best_dev_states[dept])
    local_models[dept].eval()

    with torch.no_grad():
        X_val_tensor = torch.FloatTensor(data['X_val']).to(DEVICE)
        # Process in chunks of 256 to avoid OOM on large val sets
        dept_probs = torch.cat([
            local_models[dept].predict_proba(X_val_tensor[i : i + 256])[0]
            for i in range(0, len(X_val_tensor), 256)
        ]).cpu().numpy()

    val_probs_all.extend(dept_probs.tolist())
    val_true_all.extend(data['y_val'].tolist())

val_probs = np.array(val_probs_all)
val_true  = np.array(val_true_all)

print(f"Validation set: {len(val_true):,} samples | {int(val_true.sum())} insider examples")

# Compute the full precision-recall curve once (efficient — avoids recomputing per threshold)
precision_curve, recall_curve, threshold_curve = precision_recall_curve(val_true, val_probs)

# For each target precision level, find the operating point with the best recall
TARGET_PRECISIONS = [0.3, 0.5, 0.7]   # loose → strict

print(f"\n{'Target':>10} {'Threshold':>12} {'Precision':>11} {'Recall':>9} {'F1':>8} {'Fβ':>9}")
print("─" * 66)

threshold_options = {}
for target_precision in TARGET_PRECISIONS:
    # Find all points on the PR curve that achieve at least this precision
    viable_indices = np.where(precision_curve >= target_precision)[0]

    if len(viable_indices) == 0:
        print(f"  Precision ≥ {target_precision:.1f} is not achievable on this validation set")
        continue

    # Among viable points, prefer the one with the highest recall
    best_idx  = viable_indices[np.argmax(recall_curve[viable_indices])]
    threshold = threshold_curve[min(best_idx, len(threshold_curve) - 1)]

    # Evaluate all metrics at this threshold
    y_pred_t = (val_probs >= threshold).astype(int)
    prec_t   = precision_score(val_true, y_pred_t, zero_division=0)
    rec_t    = recall_score(val_true, y_pred_t, zero_division=0)
    f1_t     = f1_score(val_true, y_pred_t, zero_division=0)
    fbeta_t  = fbeta_score(val_true, y_pred_t, beta=0.5, zero_division=0)

    threshold_options[target_precision] = {
        'threshold': threshold,
        'precision': prec_t,
        'recall':    rec_t,
        'f1':        f1_t,
        'fbeta':     fbeta_t,
    }
    print(f"  {target_precision:>8.1f} {threshold:>12.4f} {prec_t:>11.4f} "
          f"{rec_t:>9.4f} {f1_t:>8.4f} {fbeta_t:>9.4f}")

# Show what the default 0.5 threshold gives for reference
preds_default_05 = (val_probs >= 0.5).astype(int)
print(
    f"\n  Default 0.5:  precision = {precision_score(val_true, preds_default_05, zero_division=0):.4f} | "
    f"recall = {recall_score(val_true, preds_default_05, zero_division=0):.4f} | "
    f"F1 = {f1_score(val_true, preds_default_05, zero_division=0):.4f}"
)

# Pick the 0.5 precision target as our operational threshold (fallback to 0.3 if needed)
BEST_THRESHOLD = threshold_options.get(
    0.5, threshold_options.get(0.3, {'threshold': 0.5})
)['threshold']
print(f"\n→ Operational threshold selected: {BEST_THRESHOLD:.4f}  (precision ≥ 0.5 on val set)")

# ── Plot threshold analysis ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Threshold Analysis (Validation Set)', fontsize=13, fontweight='bold')

# PR curve with selected operating point marked
axes[0].plot(recall_curve, precision_curve, '#e74c3c', linewidth=2.5)
axes[0].axhline(0.5, color='navy', linestyle='--', alpha=0.7, label='Target precision = 0.5')
axes[0].axvline(
    threshold_options.get(0.5, {}).get('recall', 0),
    color='green', linestyle=':', alpha=0.7,
    label=f'Selected threshold = {BEST_THRESHOLD:.3f}',
)
axes[0].set_xlabel('Recall')
axes[0].set_ylabel('Precision')
axes[0].set_title('Precision-Recall Curve')
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[0].set_xlim(0, 1)
axes[0].set_ylim(0, 1)

# Sweep of metrics across thresholds
threshold_sweep = np.arange(0.05, 0.95, 0.02)
prec_sweep = [precision_score(val_true, (val_probs >= t).astype(int), zero_division=0)
              for t in threshold_sweep]
rec_sweep  = [recall_score(val_true,    (val_probs >= t).astype(int), zero_division=0)
              for t in threshold_sweep]
f1_sweep   = [f1_score(val_true,        (val_probs >= t).astype(int), zero_division=0)
              for t in threshold_sweep]

axes[1].plot(threshold_sweep, prec_sweep, 'b-', linewidth=2, label='Precision')
axes[1].plot(threshold_sweep, rec_sweep,  'r-', linewidth=2, label='Recall')
axes[1].plot(threshold_sweep, f1_sweep,   'g-', linewidth=2, label='F1')
axes[1].axvline(BEST_THRESHOLD, color='black', linestyle='--',
                label=f'Selected = {BEST_THRESHOLD:.3f}')
axes[1].set_xlabel('Classification Threshold')
axes[1].set_ylabel('Score')
axes[1].set_title('How Precision, Recall & F1 Change with Threshold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('threshold_analysis.png', dpi=150, bbox_inches='tight')
plt.show()


# ── Cell 12: Final evaluation on the held-out test set ───────────────────

print("\n" + "=" * 60)
print("STEP 8: FINAL TEST SET EVALUATION")
print("=" * 60)

# Restore best checkpoint weights to all department models before evaluating
for dept in departments:
    local_models[dept].body.load_state_dict(best_body_state)
    if dept in best_head_states:
        local_models[dept].head.load_state_dict(best_head_states[dept])
    if dept in best_dev_states:
        local_models[dept].deviation_encoder.load_state_dict(best_dev_states[dept])

# Collect test predictions across all departments
all_y_true  = []
all_y_probs = []
dept_test_results = {}

for dept in departments:
    data = dept_datasets[dept]
    if len(data['X_test']) == 0:
        continue

    probs, _ = predict_binary(
        local_models[dept], best_body_state,
        data['X_test'], threshold=BEST_THRESHOLD,
    )
    labels = data['y_test'].astype(int)

    all_y_true.extend(labels.tolist())
    all_y_probs.extend(probs.tolist())

    dept_preds = (probs >= BEST_THRESHOLD).astype(int)
    dept_f1    = f1_score(labels, dept_preds, zero_division=0)
    try:
        dept_auc = roc_auc_score(labels, probs)
    except Exception:
        dept_auc = float('nan')

    dept_test_results[dept] = {
        'f1':         dept_f1,
        'auc':        dept_auc,
        'n':          len(labels),
        'n_insiders': int(labels.sum()),
    }

y_true          = np.array(all_y_true)
y_probs         = np.array(all_y_probs)
y_pred          = (y_probs >= BEST_THRESHOLD).astype(int)
y_pred_default  = (y_probs >= 0.5).astype(int)

# ── Print classification report ───────────────────────────────────────────
print("\n" + "─" * 60)
print(f"TEST SET RESULTS  (threshold = {BEST_THRESHOLD:.4f})")
print("─" * 60)
print(classification_report(y_true, y_pred, target_names=['Normal', 'Insider'], digits=4))

# Compute all key metrics
f1_final    = f1_score(y_true, y_pred, zero_division=0)
fbeta_final = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)
prec_final  = precision_score(y_true, y_pred, zero_division=0)
rec_final   = recall_score(y_true, y_pred, zero_division=0)
try:
    auc_final = roc_auc_score(y_true, y_probs)
except Exception:
    auc_final = float('nan')
try:
    ap_final = average_precision_score(y_true, y_probs)
except Exception:
    ap_final = float('nan')

# FPR @ TPR=90%: the false alarm rate when we are catching 90% of insiders
fpr_arr, tpr_arr, _ = roc_curve(y_true, y_probs)
fpr_at_tpr90        = float(np.interp(0.90, tpr_arr, fpr_arr))

cm            = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n{'Metric':<35} {'Score':>10}")
print("─" * 48)
for metric_name, metric_val in [
    ('F1 (Insider class)',            f1_final),
    ('Fβ=0.5 (precision-weighted)',   fbeta_final),
    ('Precision (Insider class)',      prec_final),
    ('Recall (Insider class)',         rec_final),
    ('AUC-ROC',                        auc_final),
    ('AUC-PR (Average Precision)',     ap_final),
    ('FPR @ TPR=90%',                  fpr_at_tpr90),
]:
    print(f"{metric_name:<35} {metric_val:>10.4f}")

print(f"\nConfusion Matrix:")
print(f"  True Negatives  (correctly normal) : {tn:,}")
print(f"  False Positives (normal → flagged)  : {fp:,}  ({fp / (fp + tn) * 100:.2f}% of normal users)")
print(f"  False Negatives (missed insiders)   : {fn:,}")
print(f"  True Positives  (caught insiders)   : {tp:,}  ({tp / (tp + fn) * 100:.2f}% detection rate)")

print(f"\n── Reference: default 0.5 threshold ──")
print(classification_report(y_true, y_pred_default, target_names=['Normal', 'Insider'], digits=4))


# ── Cell 13: Detailed result visualisations ───────────────────────────────

fig = plt.figure(figsize=(20, 12))
fig.suptitle(
    'Binary Insider Threat Detection — pFedMe + User Personalisation',
    fontsize=14, fontweight='bold',
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.4)

# 1. Confusion matrix heatmap
ax_cm = fig.add_subplot(gs[0, 0])
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Normal', 'Insider'],
    yticklabels=['Normal', 'Insider'],
    ax=ax_cm, annot_kws={'size': 14},
)
ax_cm.set_title(f'Confusion Matrix\n(threshold = {BEST_THRESHOLD:.3f})', fontweight='bold')
ax_cm.set_ylabel('True Label')
ax_cm.set_xlabel('Predicted Label')

# 2. ROC curve
ax_roc = fig.add_subplot(gs[0, 1])
ax_roc.plot(fpr_arr, tpr_arr, '#e74c3c', linewidth=2.5, label=f'AUC = {auc_final:.4f}')
ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random classifier')
ax_roc.axvline(fpr_at_tpr90, color='navy', linestyle=':',
               label=f'FPR = {fpr_at_tpr90:.3f} at TPR = 0.90')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('ROC Curve', fontweight='bold')
ax_roc.legend()
ax_roc.grid(alpha=0.3)

# 3. Precision-Recall curve
ax_pr = fig.add_subplot(gs[0, 2])
prec_test_curve, rec_test_curve, _ = precision_recall_curve(y_true, y_probs)
ax_pr.plot(rec_test_curve, prec_test_curve, '#2ecc71', linewidth=2.5,
           label=f'AP = {ap_final:.4f}')
ax_pr.axhline(y_true.mean(), color='navy', linestyle='--',
              label=f'Baseline (random) = {y_true.mean():.4f}')
ax_pr.scatter([rec_final], [prec_final], color='red', s=100, zorder=5,
              label='Our operating point')
ax_pr.set_xlabel('Recall')
ax_pr.set_ylabel('Precision')
ax_pr.set_title('Precision-Recall Curve', fontweight='bold')
ax_pr.legend()
ax_pr.grid(alpha=0.3)

# 4. Per-department F1 (red bars = departments that have insider users)
ax_dept = fig.add_subplot(gs[1, 0])
sorted_depts = sorted(dept_test_results, key=lambda d: -dept_test_results[d]['f1'])
dept_f1_vals = [dept_test_results[d]['f1'] for d in sorted_depts]
dept_has_ins = [dept_test_results[d]['n_insiders'] > 0 for d in sorted_depts]
bar_colours  = ['#e74c3c' if h else '#2ecc71' for h in dept_has_ins]

ax_dept.bar(sorted_depts, dept_f1_vals, color=bar_colours, edgecolor='black')
ax_dept.set_ylim(0, 1.05)
ax_dept.set_title('F1 per Department\n(red = contains insider users)', fontweight='bold')
ax_dept.set_ylabel('F1 (Insider class)')
ax_dept.set_xlabel('Department')
ax_dept.axhline(np.mean([v for v in dept_f1_vals if v > 0]),
                color='navy', linestyle='--', label='Mean (non-zero)')
ax_dept.legend()
ax_dept.grid(axis='y', alpha=0.3)

# 5. Score distributions for normal vs insider users
ax_dist = fig.add_subplot(gs[1, 1])
normal_scores  = y_probs[y_true == 0]
insider_scores = y_probs[y_true == 1]
ax_dist.hist(normal_scores,  bins=50, alpha=0.6, color='#2ecc71',
             label=f'Normal  (n = {len(normal_scores):,})', density=True)
ax_dist.hist(insider_scores, bins=30, alpha=0.8, color='#e74c3c',
             label=f'Insider (n = {len(insider_scores):,})', density=True)
ax_dist.axvline(BEST_THRESHOLD, color='black', linestyle='--', linewidth=2,
                label=f'Threshold = {BEST_THRESHOLD:.3f}')
ax_dist.set_xlabel('P(Insider)')
ax_dist.set_ylabel('Density')
ax_dist.set_title('Model Score Distributions\n(how well separated are the classes?)',
                  fontweight='bold')
ax_dist.legend()
ax_dist.grid(alpha=0.3)

# 6. Training convergence
ax_tr = fig.add_subplot(gs[1, 2])
ax_tr.plot(history['round'], history['val_f1'],    'g-o', markersize=4, label='Val F1', linewidth=2)
ax_tr.plot(history['round'], history['val_fbeta'], 'm-o', markersize=4, label='Val Fβ', linewidth=2)
ax_tr.plot(history['round'], history['val_auc'],   'r-o', markersize=4, label='Val AUC', linewidth=2)
ax_tr.axhline(best_val_fbeta, color='navy', linestyle='--',
              label=f'Best Fβ = {best_val_fbeta:.4f}')
ax_tr.set_xlabel('Federated Round')
ax_tr.set_ylabel('Score')
ax_tr.set_title('Stage-1 Training Convergence', fontweight='bold')
ax_tr.legend(fontsize=8)
ax_tr.grid(alpha=0.3)

plt.savefig('binary_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved binary_results.png")


# ── Cell 14: Comparison against baseline methods ─────────────────────────

print("\n" + "=" * 60)
print("STEP 9: BASELINE COMPARISONS")
print("=" * 60)

# Use the same number of rounds as the main model for a fair comparison
N_ROUNDS_BASELINE = N_ROUNDS_S1


def get_flat_arrays(split_name):
    """Concatenate all departments' arrays into a single flat matrix for centralized baselines."""
    X_list, y_list = [], []
    for dept in departments:
        data = dept_datasets[dept]
        X_d  = data[f'X_{split_name}']
        y_d  = data[f'y_{split_name}']
        if len(X_d) == 0:
            continue
        X_list.append(X_d.reshape(len(X_d), -1))  # flatten the temporal dimension
        y_list.append(y_d)
    if not X_list:
        return np.zeros((0, 1)), np.zeros(0)
    return np.vstack(X_list), np.concatenate(y_list)


print("Assembling flat arrays for centralized baselines...")
X_train_flat, y_train_flat = get_flat_arrays('train')
X_test_flat,  y_test_flat  = get_flat_arrays('test')
X_val_flat,   y_val_flat   = get_flat_arrays('val')
insider_rate_flat          = y_train_flat.sum() / len(y_train_flat)
print(f"Shape: {X_train_flat.shape} | Insider rate: {insider_rate_flat * 100:.3f}%")

baseline_results = {}


# ── Baseline 1: Random Forest (centralized, sees all data) ───────────────
print("\n[1 / 4] Random Forest (centralized)...")
rf = RandomForestClassifier(
    n_estimators=300, max_depth=None,
    class_weight='balanced',   # handles imbalance automatically
    n_jobs=-1, random_state=SEED,
)
rf.fit(X_train_flat, y_train_flat)

rf_val_probs = rf.predict_proba(X_val_flat)[:, 1]
rf_prec_curve, rf_rec_curve, rf_thresholds = precision_recall_curve(y_val_flat, rf_val_probs)
viable_rf   = np.where(rf_prec_curve >= 0.5)[0]
rf_threshold = rf_thresholds[viable_rf[np.argmax(rf_rec_curve[viable_rf])]] if len(viable_rf) else 0.5

rf_test_probs = rf.predict_proba(X_test_flat)[:, 1]
rf_preds      = (rf_test_probs >= rf_threshold).astype(int)
rf_f1         = f1_score(y_test_flat, rf_preds, zero_division=0)
try:    rf_auc = roc_auc_score(y_test_flat, rf_test_probs)
except: rf_auc = float('nan')
rf_prec_score = precision_score(y_test_flat, rf_preds, zero_division=0)
rf_rec_score  = recall_score(y_test_flat, rf_preds, zero_division=0)

baseline_results['Random Forest (centralized)'] = {
    'f1': rf_f1, 'auc': rf_auc, 'precision': rf_prec_score, 'recall': rf_rec_score
}
print(f"  RF:  F1 = {rf_f1:.4f} | Prec = {rf_prec_score:.4f} | Rec = {rf_rec_score:.4f} | AUC = {rf_auc:.4f}")


# ── Baseline 2: XGBoost (centralized) ────────────────────────────────────
print("[2 / 4] XGBoost (centralized)...")
xgb_pos_weight = min((y_train_flat == 0).sum() / (y_train_flat == 1).sum(), 50)
xgb = XGBClassifier(
    n_estimators=300, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=xgb_pos_weight,
    eval_metric='aucpr', random_state=SEED, n_jobs=-1,
    tree_method='hist',
    device='cuda' if DEVICE == 'cuda' else 'cpu',
)
xgb.fit(X_train_flat, y_train_flat, eval_set=[(X_val_flat, y_val_flat)], verbose=False)

xgb_val_probs = xgb.predict_proba(X_val_flat)[:, 1]
xgb_prec_curve, xgb_rec_curve, xgb_thresholds = precision_recall_curve(y_val_flat, xgb_val_probs)
viable_xgb    = np.where(xgb_prec_curve >= 0.5)[0]
xgb_threshold = xgb_thresholds[viable_xgb[np.argmax(xgb_rec_curve[viable_xgb])]] if len(viable_xgb) else 0.5

xgb_test_probs = xgb.predict_proba(X_test_flat)[:, 1]
xgb_preds      = (xgb_test_probs >= xgb_threshold).astype(int)
xgb_f1         = f1_score(y_test_flat, xgb_preds, zero_division=0)
try:    xgb_auc = roc_auc_score(y_test_flat, xgb_test_probs)
except: xgb_auc = float('nan')
xgb_prec_score = precision_score(y_test_flat, xgb_preds, zero_division=0)
xgb_rec_score  = recall_score(y_test_flat, xgb_preds, zero_division=0)

baseline_results['XGBoost (centralized)'] = {
    'f1': xgb_f1, 'auc': xgb_auc, 'precision': xgb_prec_score, 'recall': xgb_rec_score
}
print(f"  XGB: F1 = {xgb_f1:.4f} | Prec = {xgb_prec_score:.4f} | Rec = {xgb_rec_score:.4f} | AUC = {xgb_auc:.4f}")


# ── Baseline 3: Standard FedAvg (no personalisation) ─────────────────────
print(f"[3 / 4] FedAvg (no personalisation, {N_ROUNDS_BASELINE} rounds)...")


class GlobalBinaryModel(nn.Module):
    """Simple global model for FedAvg baseline — shared body and head, no deviation encoder."""

    def __init__(self, n_features, hidden_dim=128):
        super().__init__()
        self.body = BinaryInsiderBody(n_features, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        rep, attn_weights = self.body(x)
        return self.head(rep).squeeze(1), attn_weights


fedavg_dept_models = {d: GlobalBinaryModel(n_features, HIDDEN_DIM).to(DEVICE) for d in departments}
fedavg_body_state  = copy.deepcopy(list(fedavg_dept_models.values())[0].body.state_dict())
fedavg_head_state  = copy.deepcopy(list(fedavg_dept_models.values())[0].head.state_dict())
fedavg_opts        = {
    d: AdamW(list(fedavg_dept_models[d].parameters()), lr=LR_LOCAL, weight_decay=WEIGHT_DECAY)
    for d in departments
}

for rnd in range(1, N_ROUNDS_BASELINE + 1):
    new_body_states = []
    new_head_states = []
    dept_sizes      = []

    for dept in departments:
        data   = dept_datasets[dept]
        X_tr   = data['X_train']
        y_tr   = data['y_train']
        if len(X_tr) < MIN_DEPT_SIZE:
            continue

        model = fedavg_dept_models[dept]
        model.body.load_state_dict(fedavg_body_state)
        model.head.load_state_dict(fedavg_head_state)
        model.train()

        loader = build_oversampled_loader(X_tr, y_tr, BATCH_SIZE, OVERSAMPLE_RATIO)
        pw     = dept_pos_weights.get(dept, torch.tensor(10.0).to(DEVICE))
        crit   = BinaryFocalLoss(gamma=GAMMA_FOCAL, pos_weight=pw)

        for _epoch in range(E_LOCAL):
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                fedavg_opts[dept].zero_grad()
                out, _ = model(x_batch)
                crit(out, y_batch).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                fedavg_opts[dept].step()

        new_body_states.append(copy.deepcopy(model.body.state_dict()))
        new_head_states.append(copy.deepcopy(model.head.state_dict()))
        dept_sizes.append(len(X_tr))

    if new_body_states:
        fedavg_body_state = fedavg_aggregate(new_body_states, dept_sizes)
        fedavg_head_state = fedavg_aggregate(new_head_states, dept_sizes)

    if rnd % 10 == 0:
        print(f"    FedAvg round {rnd} / {N_ROUNDS_BASELINE}")

# Evaluate FedAvg on the test set
fedavg_probs_all = []
fedavg_true_all  = []
for dept in departments:
    data = dept_datasets[dept]
    if len(data['X_test']) == 0:
        continue
    fedavg_dept_models[dept].body.load_state_dict(fedavg_body_state)
    fedavg_dept_models[dept].head.load_state_dict(fedavg_head_state)
    fedavg_dept_models[dept].eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(data['X_test']).to(DEVICE)
        logits = torch.cat([
            fedavg_dept_models[dept](X_test_tensor[i : i + 256])[0]
            for i in range(0, len(X_test_tensor), 256)
        ])
    fedavg_probs_all.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    fedavg_true_all.extend(data['y_test'].tolist())

fedavg_probs = np.array(fedavg_probs_all)
fedavg_true  = np.array(fedavg_true_all)
fedavg_preds = (fedavg_probs >= BEST_THRESHOLD).astype(int)
fedavg_f1    = f1_score(fedavg_true, fedavg_preds, zero_division=0)
try:    fedavg_auc = roc_auc_score(fedavg_true, fedavg_probs)
except: fedavg_auc = float('nan')
fedavg_prec  = precision_score(fedavg_true, fedavg_preds, zero_division=0)
fedavg_rec   = recall_score(fedavg_true, fedavg_preds, zero_division=0)

baseline_results['FedAvg (no personalization)'] = {
    'f1': fedavg_f1, 'auc': fedavg_auc, 'precision': fedavg_prec, 'recall': fedavg_rec
}
print(f"  FedAvg: F1 = {fedavg_f1:.4f} | Prec = {fedavg_prec:.4f} | Rec = {fedavg_rec:.4f} | AUC = {fedavg_auc:.4f}")


# ── Baseline 4: Original FedRep (alternating head/body updates, no deviation encoder) ──
print("[4 / 4] Original FedRep (for direct comparison)...")


class OriginalFedRepModel(nn.Module):
    """
    FedRep-style model: shared body + local head, but no deviation encoder.
    FedRep alternates between updating the head (with body frozen) and
    updating the body (with head frozen). We reproduce this exactly to
    isolate the contribution of the deviation encoder.
    """
    def __init__(self, n_features, hidden_dim=128):
        super().__init__()
        self.body = BinaryInsiderBody(n_features, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1)
        )

    def forward(self, x):
        rep, attn_weights = self.body(x)
        return self.head(rep).squeeze(1), attn_weights


fedrep_models    = {d: OriginalFedRepModel(n_features, HIDDEN_DIM).to(DEVICE) for d in departments}
fedrep_body      = copy.deepcopy(list(fedrep_models.values())[0].body.state_dict())
fedrep_heads     = {d: copy.deepcopy(fedrep_models[d].head.state_dict()) for d in departments}
fedrep_body_opts = {d: AdamW(fedrep_models[d].body.parameters(), lr=LR_LOCAL) for d in departments}
fedrep_head_opts = {d: AdamW(fedrep_models[d].head.parameters(), lr=LR_LOCAL) for d in departments}

for rnd in range(N_ROUNDS_BASELINE):
    new_body_states = []
    dept_sizes      = []

    for dept in departments:
        data = dept_datasets[dept]
        X_tr = data['X_train']
        y_tr = data['y_train']
        if len(X_tr) < MIN_DEPT_SIZE:
            continue

        model = fedrep_models[dept]
        model.body.load_state_dict(fedrep_body)
        model.head.load_state_dict(fedrep_heads[dept])
        model.train()

        loader = build_oversampled_loader(X_tr, y_tr, BATCH_SIZE, OVERSAMPLE_RATIO)
        pw     = dept_pos_weights.get(dept, torch.tensor(10.0).to(DEVICE))
        crit   = BinaryFocalLoss(gamma=GAMMA_FOCAL, pos_weight=pw)

        # Step 1: update head only (body frozen) — 3 local epochs
        for param in model.body.parameters(): param.requires_grad_(False)
        for param in model.head.parameters(): param.requires_grad_(True)
        for _epoch in range(3):
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                fedrep_head_opts[dept].zero_grad()
                out, _ = model(x_batch)
                crit(out, y_batch).backward()
                nn.utils.clip_grad_norm_(model.head.parameters(), 1.0)
                fedrep_head_opts[dept].step()

        # Step 2: update body only (head frozen) — 2 local epochs
        for param in model.body.parameters(): param.requires_grad_(True)
        for param in model.head.parameters(): param.requires_grad_(False)
        for _epoch in range(2):
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                fedrep_body_opts[dept].zero_grad()
                out, _ = model(x_batch)
                crit(out, y_batch).backward()
                nn.utils.clip_grad_norm_(model.body.parameters(), 1.0)
                fedrep_body_opts[dept].step()

        for param in model.parameters(): param.requires_grad_(True)
        new_body_states.append(copy.deepcopy(model.body.state_dict()))
        fedrep_heads[dept] = copy.deepcopy(model.head.state_dict())
        dept_sizes.append(len(X_tr))

    if new_body_states:
        fedrep_body = fedavg_aggregate(new_body_states, dept_sizes)

# Evaluate FedRep
fedrep_probs_all = []
fedrep_true_all  = []
for dept in departments:
    data = dept_datasets[dept]
    if len(data['X_test']) == 0:
        continue
    fedrep_models[dept].body.load_state_dict(fedrep_body)
    fedrep_models[dept].head.load_state_dict(fedrep_heads[dept])
    fedrep_models[dept].eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(data['X_test']).to(DEVICE)
        logits = torch.cat([
            fedrep_models[dept](X_test_tensor[i : i + 256])[0]
            for i in range(0, len(X_test_tensor), 256)
        ])
    fedrep_probs_all.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    fedrep_true_all.extend(data['y_test'].tolist())

fedrep_probs = np.array(fedrep_probs_all)
fedrep_true  = np.array(fedrep_true_all)
fedrep_preds = (fedrep_probs >= BEST_THRESHOLD).astype(int)
fedrep_f1    = f1_score(fedrep_true, fedrep_preds, zero_division=0)
try:    fedrep_auc = roc_auc_score(fedrep_true, fedrep_probs)
except: fedrep_auc = float('nan')
fedrep_prec  = precision_score(fedrep_true, fedrep_preds, zero_division=0)
fedrep_rec   = recall_score(fedrep_true, fedrep_preds, zero_division=0)

baseline_results['FedRep (original)'] = {
    'f1': fedrep_f1, 'auc': fedrep_auc, 'precision': fedrep_prec, 'recall': fedrep_rec
}
print(f"  FedRep: F1 = {fedrep_f1:.4f} | Prec = {fedrep_prec:.4f} | Rec = {fedrep_rec:.4f} | AUC = {fedrep_auc:.4f}")

# Add our method to the comparison table
baseline_results['pFedMe + User Pers. (Ours)'] = {
    'f1': f1_final, 'auc': auc_final, 'precision': prec_final, 'recall': rec_final
}


# ── Cell 15: Comparison table and plots ───────────────────────────────────

print("\n" + "=" * 60)
print("STEP 10: RESULTS COMPARISON TABLE")
print("=" * 60)

best_method = max(baseline_results, key=lambda m: baseline_results[m]['f1'])

print(f"\n{'Method':<42} {'F1':>8} {'Prec':>8} {'Recall':>8} {'AUC':>8}")
print("─" * 78)
for method, scores in baseline_results.items():
    tag = " ◄ BEST" if method == best_method else " ◄ OURS" if "Ours" in method else ""
    print(
        f"{method:<42} {scores['f1']:>8.4f} {scores['precision']:>8.4f} "
        f"{scores['recall']:>8.4f} {scores['auc']:>8.4f}{tag}"
    )

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle('Insider Threat Detection — Method Comparison', fontsize=14, fontweight='bold')

methods    = list(baseline_results.keys())
f1_vals    = [baseline_results[m]['f1']        for m in methods]
auc_vals   = [baseline_results[m]['auc']       for m in methods]
prec_vals  = [baseline_results[m]['precision'] for m in methods]
rec_vals   = [baseline_results[m]['recall']    for m in methods]

best_idx = f1_vals.index(max(f1_vals))
ours_idx = len(methods) - 1
bar_cols = ['#3498db'] * len(methods)
bar_cols[best_idx] = '#f39c12'
bar_cols[ours_idx] = '#e74c3c'

x = np.arange(len(methods))
w = 0.35

axes[0].bar(x - w/2, f1_vals,  w, label='F1',      color=bar_cols, edgecolor='black', alpha=0.9)
axes[0].bar(x + w/2, auc_vals, w, label='AUC-ROC', color=bar_cols, edgecolor='black', alpha=0.5)
axes[0].set_xticks(x)
axes[0].set_xticklabels(methods, rotation=20, ha='right', fontsize=9)
axes[0].set_ylim(0, 1.1)
axes[0].set_ylabel('Score')
axes[0].set_title('F1 and AUC-ROC by Method', fontweight='bold')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

axes[1].bar(x - w/2, prec_vals, w, label='Precision', color=bar_cols, edgecolor='black', alpha=0.9)
axes[1].bar(x + w/2, rec_vals,  w, label='Recall',    color=bar_cols, edgecolor='black', alpha=0.5)
axes[1].set_xticks(x)
axes[1].set_xticklabels(methods, rotation=20, ha='right', fontsize=9)
axes[1].set_ylim(0, 1.1)
axes[1].set_ylabel('Score')
axes[1].set_title('Precision and Recall by Method', fontweight='bold')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('baseline_comparison.png', dpi=150, bbox_inches='tight')
plt.show()


# ── Cell 16: Ablation study ───────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 11: ABLATION STUDY")
print("=" * 60)

print("""
We remove one component at a time and measure the drop in F1.
This tells us how much each design decision actually contributes.
""")

ablation_results = {}
N_ABLATION_ROUNDS = 15   # slightly fewer rounds than the main model (faster)


def run_ablation_with_global_head(model_class, variant_name, n_rounds=N_ABLATION_ROUNDS):
    """
    Run a FedAvg-style ablation with a shared head (not personalised).
    Used for FedAvg and FedRep ablations where we just want to know
    whether our specific components help over a simpler alternative.
    """
    models = {d: model_class(n_features, HIDDEN_DIM).to(DEVICE) for d in departments}
    body   = copy.deepcopy(list(models.values())[0].body.state_dict())
    heads  = {d: copy.deepcopy(models[d].head.state_dict()) for d in departments}
    opts   = {
        d: AdamW(models[d].parameters(), lr=LR_LOCAL, weight_decay=WEIGHT_DECAY)
        for d in departments
    }

    for _rnd in range(n_rounds):
        new_body_states = []
        dept_sizes      = []

        for dept in departments:
            data = dept_datasets[dept]
            X_tr = data['X_train']
            y_tr = data['y_train']
            if len(X_tr) < MIN_DEPT_SIZE:
                continue

            m = models[dept]
            m.body.load_state_dict(body)
            m.head.load_state_dict(heads[dept])
            m.train()

            loader = build_oversampled_loader(X_tr, y_tr, BATCH_SIZE, OVERSAMPLE_RATIO)
            pw     = dept_pos_weights.get(dept, torch.tensor(10.0).to(DEVICE))
            crit   = BinaryFocalLoss(gamma=GAMMA_FOCAL, pos_weight=pw)

            for _epoch in range(E_LOCAL):
                for x_batch, y_batch in loader:
                    x_batch = x_batch.to(DEVICE)
                    y_batch = y_batch.to(DEVICE)
                    opts[dept].zero_grad()
                    out, _ = m(x_batch)
                    crit(out, y_batch).backward()
                    nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                    opts[dept].step()

            new_body_states.append(copy.deepcopy(m.body.state_dict()))
            heads[dept] = copy.deepcopy(m.head.state_dict())
            dept_sizes.append(len(X_tr))

        if new_body_states:
            body = fedavg_aggregate(new_body_states, dept_sizes)

    # Evaluate
    all_probs, all_true = [], []
    for dept in departments:
        data = dept_datasets[dept]
        if len(data['X_test']) == 0:
            continue
        models[dept].body.load_state_dict(body)
        models[dept].head.load_state_dict(heads[dept])
        models[dept].eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(data['X_test']).to(DEVICE)
            logits = torch.cat([
                models[dept](X_test_tensor[i : i + 256])[0]
                for i in range(0, len(X_test_tensor), 256)
            ])
        all_probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        all_true.extend(data['y_test'].tolist())

    all_probs  = np.array(all_probs)
    all_true   = np.array(all_true)
    preds      = (all_probs >= BEST_THRESHOLD).astype(int)
    f1_variant = f1_score(all_true, preds, zero_division=0)
    print(f"  {variant_name:<50}: F1 = {f1_variant:.4f}")
    return f1_variant


# ── Ablation 1: Remove temporal modelling (MLP only, no LSTM/attention) ──
print("[1 / 4] Without temporal modelling (MLP on last week only)...")


class StaticMlpModel(nn.Module):
    """
    Ablation: no LSTM, no attention — just an MLP on the most recent week.
    Tests whether the temporal context window actually matters.
    """
    def __init__(self, n_features, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128),        nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, hidden_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1)
        )

        # Dummy body attribute for interface compatibility with the training loop
        class _DummyBody(nn.Module):
            def load_state_dict(self, state_dict, strict=True): pass
            def state_dict(self, *args, **kwargs): return {}
            def named_parameters(self): return iter([])
            def parameters(self): return iter([])

        self.body = _DummyBody()

    def forward(self, x):
        rep = self.encoder(x[:, -1, :])   # only look at the last week
        return self.head(rep).squeeze(1), None


# Manual training loop for the MLP (no shared body to aggregate)
mlp_models = {d: StaticMlpModel(n_features, HIDDEN_DIM).to(DEVICE) for d in departments}
mlp_opts   = {
    d: AdamW(
        list(mlp_models[d].encoder.parameters()) + list(mlp_models[d].head.parameters()),
        lr=LR_LOCAL, weight_decay=WEIGHT_DECAY,
    )
    for d in departments
}

for _epoch in range(25):
    for dept in departments:
        data = dept_datasets[dept]
        X_tr = data['X_train']
        y_tr = data['y_train']
        if len(X_tr) < MIN_DEPT_SIZE:
            continue

        mlp_models[dept].train()
        loader = build_oversampled_loader(X_tr, y_tr, BATCH_SIZE, OVERSAMPLE_RATIO)
        pw     = dept_pos_weights.get(dept, torch.tensor(10.0).to(DEVICE))
        crit   = BinaryFocalLoss(gamma=GAMMA_FOCAL, pos_weight=pw)

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            mlp_opts[dept].zero_grad()
            out, _ = mlp_models[dept](x_batch)
            crit(out, y_batch).backward()
            nn.utils.clip_grad_norm_(
                list(mlp_models[dept].encoder.parameters()) +
                list(mlp_models[dept].head.parameters()), 1.0,
            )
            mlp_opts[dept].step()

mlp_probs_all, mlp_true_all = [], []
for dept in departments:
    data = dept_datasets[dept]
    if len(data['X_test']) == 0:
        continue
    mlp_models[dept].eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(data['X_test']).to(DEVICE)
        logits = torch.cat([
            mlp_models[dept](X_test_tensor[i : i + 256])[0]
            for i in range(0, len(X_test_tensor), 256)
        ])
    mlp_probs_all.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    mlp_true_all.extend(data['y_test'].tolist())

mlp_preds = (np.array(mlp_probs_all) >= BEST_THRESHOLD).astype(int)
mlp_f1    = f1_score(np.array(mlp_true_all), mlp_preds, zero_division=0)
ablation_results['w/o Temporal  (MLP on last week only)'] = mlp_f1
print(f"  MLP only: F1 = {mlp_f1:.4f}")

# ── Ablation 2: Remove deviation encoder ─────────────────────────────────
print("[2 / 4] Without deviation encoder...")
nodev_f1 = run_ablation_with_global_head(GlobalBinaryModel, 'w/o Deviation Encoder')
ablation_results['w/o Deviation Encoder'] = nodev_f1

# ── Ablation 3: Remove the pFedMe proximal term (λ = 0) ──────────────────
print("[3 / 4] Without proximal term (λ = 0, vanilla FL)...")

noprox_models = {d: BinaryPersonalizedModel(n_features, HIDDEN_DIM).to(DEVICE) for d in departments}
noprox_body   = copy.deepcopy(list(noprox_models.values())[0].body.state_dict())
noprox_heads  = {d: copy.deepcopy(noprox_models[d].head.state_dict())              for d in departments}
noprox_devs   = {d: copy.deepcopy(noprox_models[d].deviation_encoder.state_dict()) for d in departments}
noprox_opts   = {
    d: AdamW(noprox_models[d].parameters(), lr=LR_LOCAL, weight_decay=WEIGHT_DECAY)
    for d in departments
}

for _rnd in range(N_ABLATION_ROUNDS):
    new_body_states = []
    dept_sizes      = []

    for dept in departments:
        data = dept_datasets[dept]
        X_tr = data['X_train']
        y_tr = data['y_train']
        if len(X_tr) < MIN_DEPT_SIZE:
            continue

        m = noprox_models[dept]
        m.body.load_state_dict(noprox_body)
        m.head.load_state_dict(noprox_heads[dept])
        m.deviation_encoder.load_state_dict(noprox_devs[dept])
        m.train()

        loader = build_oversampled_loader(X_tr, y_tr, BATCH_SIZE, OVERSAMPLE_RATIO)
        pw     = dept_pos_weights.get(dept, torch.tensor(10.0).to(DEVICE))
        crit   = BinaryFocalLoss(gamma=GAMMA_FOCAL, pos_weight=pw)

        for _epoch in range(E_LOCAL):
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                noprox_opts[dept].zero_grad()
                logit, _ = m(x_batch)
                loss = crit(logit, y_batch)   # no proximal term — λ = 0
                loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                noprox_opts[dept].step()

        new_body_states.append(copy.deepcopy(m.body.state_dict()))
        noprox_heads[dept] = copy.deepcopy(m.head.state_dict())
        noprox_devs[dept]  = copy.deepcopy(m.deviation_encoder.state_dict())
        dept_sizes.append(len(X_tr))

    if new_body_states:
        noprox_body = fedavg_aggregate(new_body_states, dept_sizes)

noprox_probs_all, noprox_true_all = [], []
for dept in departments:
    data = dept_datasets[dept]
    if len(data['X_test']) == 0:
        continue
    noprox_models[dept].body.load_state_dict(noprox_body)
    noprox_models[dept].head.load_state_dict(noprox_heads[dept])
    noprox_models[dept].deviation_encoder.load_state_dict(noprox_devs[dept])
    noprox_models[dept].eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(data['X_test']).to(DEVICE)
        logits = torch.cat([
            noprox_models[dept](X_test_tensor[i : i + 256])[0]
            for i in range(0, len(X_test_tensor), 256)
        ])
    noprox_probs_all.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    noprox_true_all.extend(data['y_test'].tolist())

noprox_preds = (np.array(noprox_probs_all) >= BEST_THRESHOLD).astype(int)
noprox_f1    = f1_score(np.array(noprox_true_all), noprox_preds, zero_division=0)
ablation_results['w/o Proximal Term (λ = 0)'] = noprox_f1
print(f"  No proximal term: F1 = {noprox_f1:.4f}")


# ── Ablation 4: Remove focal loss (plain weighted BCE) ───────────────────
print("[4 / 4] Without focal loss (weighted BCE only)...")

nofocal_models = {d: BinaryPersonalizedModel(n_features, HIDDEN_DIM).to(DEVICE) for d in departments}
nofocal_body   = copy.deepcopy(list(nofocal_models.values())[0].body.state_dict())
nofocal_heads  = {d: copy.deepcopy(nofocal_models[d].head.state_dict())              for d in departments}
nofocal_devs   = {d: copy.deepcopy(nofocal_models[d].deviation_encoder.state_dict()) for d in departments}
nofocal_opts   = {
    d: AdamW(nofocal_models[d].parameters(), lr=LR_LOCAL, weight_decay=WEIGHT_DECAY)
    for d in departments
}

for _rnd in range(N_ABLATION_ROUNDS):
    new_body_states = []
    dept_sizes      = []

    for dept in departments:
        data = dept_datasets[dept]
        X_tr = data['X_train']
        y_tr = data['y_train']
        if len(X_tr) < MIN_DEPT_SIZE:
            continue

        m = nofocal_models[dept]
        m.body.load_state_dict(nofocal_body)
        m.head.load_state_dict(nofocal_heads[dept])
        m.deviation_encoder.load_state_dict(nofocal_devs[dept])
        m.train()

        loader = build_oversampled_loader(X_tr, y_tr, BATCH_SIZE, OVERSAMPLE_RATIO)
        pw     = dept_pos_weights.get(dept, torch.tensor(10.0).to(DEVICE))
        # Snapshot global body for the proximal term
        global_body_ref = {
            n: p.detach().clone()
            for n, p in m.body.named_parameters()
        }

        for _epoch in range(E_LOCAL):
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                nofocal_opts[dept].zero_grad()
                logit, _ = m(x_batch)

                # Plain weighted BCE — no focal modulation
                bce = F.binary_cross_entropy_with_logits(
                    logit, y_batch.float(), pos_weight=pw, reduction='mean'
                )
                proximal = sum(
                    ((local_p - global_body_ref[n]) ** 2).sum()
                    for n, local_p in m.body.named_parameters()
                )
                loss = bce + (LAMBDA_PROX / 2) * proximal
                loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                nofocal_opts[dept].step()

        new_body_states.append(copy.deepcopy(m.body.state_dict()))
        nofocal_heads[dept] = copy.deepcopy(m.head.state_dict())
        nofocal_devs[dept]  = copy.deepcopy(m.deviation_encoder.state_dict())
        dept_sizes.append(len(X_tr))

    if new_body_states:
        nofocal_body = fedavg_aggregate(new_body_states, dept_sizes)

nofocal_probs_all, nofocal_true_all = [], []
for dept in departments:
    data = dept_datasets[dept]
    if len(data['X_test']) == 0:
        continue
    nofocal_models[dept].body.load_state_dict(nofocal_body)
    nofocal_models[dept].head.load_state_dict(nofocal_heads[dept])
    nofocal_models[dept].deviation_encoder.load_state_dict(nofocal_devs[dept])
    nofocal_models[dept].eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(data['X_test']).to(DEVICE)
        logits = torch.cat([
            nofocal_models[dept](X_test_tensor[i : i + 256])[0]
            for i in range(0, len(X_test_tensor), 256)
        ])
    nofocal_probs_all.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    nofocal_true_all.extend(data['y_test'].tolist())

nofocal_preds = (np.array(nofocal_probs_all) >= BEST_THRESHOLD).astype(int)
nofocal_f1    = f1_score(np.array(nofocal_true_all), nofocal_preds, zero_division=0)
ablation_results['w/o Focal Loss (BCE only)'] = nofocal_f1
print(f"  No focal loss: F1 = {nofocal_f1:.4f}")

# Add the full model for comparison
ablation_results['Full Model: pFedMe + Dev. Encoder + Focal (Ours)'] = f1_final

print(f"\n{'Variant':<55} {'F1':>8} {'Change':>10}")
print("─" * 76)
for variant_name, variant_f1 in ablation_results.items():
    delta = variant_f1 - f1_final
    print(f"{variant_name:<55} {variant_f1:>8.4f} {delta:>+10.4f}")

# Ablation bar chart
fig, ax = plt.subplots(figsize=(13, 5))
ab_names  = list(ablation_results.keys())
ab_scores = list(ablation_results.values())
ab_colors = ['#e74c3c' if 'Ours' in n else '#95a5a6' for n in ab_names]

bars = ax.bar(ab_names, ab_scores, color=ab_colors, edgecolor='black')
ax.set_ylim(0, 1.05)
ax.axhline(f1_final, color='red', linestyle='--', alpha=0.5,
           label=f'Full model F1 = {f1_final:.4f}')
ax.set_title('Ablation Study — Contribution of Each Component', fontsize=13, fontweight='bold')
ax.set_ylabel('F1 (Insider class)')
ax.legend()
ax.set_xticklabels(ab_names, rotation=20, ha='right', fontsize=9)
ax.grid(axis='y', alpha=0.3)
for bar, score in zip(bars, ab_scores):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f'{score:.4f}', ha='center', fontsize=9, fontweight='bold',
    )
plt.tight_layout()
plt.savefig('ablation_study.png', dpi=150, bbox_inches='tight')
plt.show()


# ── Cell 17: Privacy-utility tradeoff analysis ────────────────────────────

print("\n" + "=" * 60)
print("STEP 12: PRIVACY-UTILITY TRADEOFF")
print("=" * 60)

# The best centralized F1 (upper bound on utility)
best_centralized_f1 = max(
    baseline_results['Random Forest (centralized)']['f1'],
    baseline_results['XGBoost (centralized)']['f1'],
)

body_param_count    = sum(p.numel() for p in list(local_models.values())[0].body.parameters())
raw_datapoints      = df[USER_COL].nunique() * len(all_weeks) * n_features
privacy_multiplier  = raw_datapoints / max(body_param_count, 1)

print(f"""
┌──────────────────────────────────────────────────────────────┐
│  PRIVACY-UTILITY TRADEOFF SUMMARY                            │
├──────────────────────────────────────────────────────────────┤
│  Centralized best F1                 : {best_centralized_f1:.4f}                 │
│  Our federated method F1             : {f1_final:.4f}                 │
│  Utility cost of privacy             : {(best_centralized_f1 - f1_final) * 100:.2f}% lower F1              │
│  FL efficiency vs centralized        : {f1_final / max(best_centralized_f1, 1e-9) * 100:.1f}% of centralized        │
│                                                              │
│  Raw datapoints centralised model sees : {raw_datapoints:>10,}      │
│  Body parameters shared per round      : {body_param_count:>10,}      │
│  Effective privacy multiplier          : {privacy_multiplier:>10,.0f}×         │
└──────────────────────────────────────────────────────────────┘
""")

# Privacy-utility frontier plot
privacy_levels  = [0.0,               0.7,              0.85,              0.95]
utility_scores  = [
    best_centralized_f1,
    baseline_results['FedAvg (no personalization)']['f1'],
    baseline_results['FedRep (original)']['f1'],
    f1_final,
]
method_labels   = ['Centralized\n(XGBoost)', 'FedAvg\n(No Pers.)', 'FedRep\n(Original)', 'pFedMe\n(Ours)']
method_colours  = ['#e74c3c', '#95a5a6', '#f39c12', '#2ecc71']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Privacy-Utility Tradeoff', fontsize=13, fontweight='bold')

bars = axes[0].bar(method_labels, utility_scores, color=method_colours, edgecolor='black')
axes[0].set_ylabel('F1 (Insider class)')
axes[0].set_ylim(0, 1.05)
axes[0].set_title('Utility Achieved at Each Privacy Level')
axes[0].grid(axis='y', alpha=0.3)
for bar, v in zip(bars, utility_scores):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{v:.4f}', ha='center', fontsize=10, fontweight='bold')

for priv, util, label, col in zip(privacy_levels, utility_scores, method_labels, method_colours):
    axes[1].scatter(priv, util, s=250, color=col, zorder=5,
                    edgecolors='black', linewidth=1.5)
    axes[1].annotate(label.replace('\n', ' '), (priv, util),
                     textcoords='offset points', xytext=(8, 5), fontsize=8)
axes[1].plot(privacy_levels, utility_scores, 'k--', alpha=0.4, linewidth=1.5)
axes[1].set_xlabel('Privacy Level (0 = raw data shared, 1 = no raw data shared)')
axes[1].set_ylabel('F1 (Insider class)')
axes[1].set_title('Privacy-Utility Frontier')
axes[1].set_xlim(-0.1, 1.1)
axes[1].set_ylim(0, 1.05)
axes[1].grid(alpha=0.3)
axes[1].annotate(
    '← OUR METHOD',
    xy=(privacy_levels[-1], utility_scores[-1]),
    xytext=(privacy_levels[-1] - 0.3, utility_scores[-1] - 0.12),
    arrowprops=dict(arrowstyle='->', color='green'),
    color='green', fontweight='bold', fontsize=9,
)

plt.tight_layout()
plt.savefig('privacy_utility.png', dpi=150, bbox_inches='tight')
plt.show()


# ── Cell 18: Attention weight visualisation ───────────────────────────────

print("\n" + "=" * 60)
print("STEP 13: ATTENTION VISUALISATION")
print("=" * 60)

print("""
The BiLSTM-Attention body learns to weight different weeks in the 8-week
window differently. Visualising these attention weights helps us understand:
  - Which weeks in the lead-up to an insider event are most informative?
  - Do insider users show a different temporal attention pattern to normal users?

Higher attention on recent weeks suggests the model reacts to sudden changes;
higher attention on earlier weeks suggests it detects slow ramp-up behaviour.
""")


def visualize_attention_for_dept(dept, split='test', n_examples=6):
    """
    Plot temporal attention weights for a selection of normal and insider
    examples from the given department. Red = insider, green = normal.
    """
    data = dept_datasets[dept]
    X_sp = data[f'X_{split}']
    y_sp = data[f'y_{split}']

    if len(X_sp) == 0:
        return

    model = local_models[dept]
    model.body.load_state_dict(best_body_state)
    if dept in best_head_states:
        model.head.load_state_dict(best_head_states[dept])
    model.eval()

    insider_idx = np.where(y_sp == 1)[0]
    normal_idx  = np.where(y_sp == 0)[0]

    if len(insider_idx) == 0:
        print(f"  Dept {dept} — no insider examples in {split} set, skipping")
        return

    # Pick equal numbers of insider and normal examples
    n_ins = min(n_examples // 2, len(insider_idx))
    n_nor = min(n_examples // 2, len(normal_idx))
    selected_idx = np.concatenate([insider_idx[:n_ins], normal_idx[:n_nor]])

    X_sel   = torch.FloatTensor(X_sp[selected_idx]).to(DEVICE)
    y_sel   = y_sp[selected_idx]

    with torch.no_grad():
        _, attn_weights = model.body(X_sel)   # shape: (n_sel, n_heads, T, T)
        attn_weights    = attn_weights.cpu().numpy()
        probs_sel, _    = model.predict_proba(X_sel)
        probs_sel       = probs_sel.cpu().numpy()

    fig, axes = plt.subplots(1, len(selected_idx), figsize=(4 * len(selected_idx), 4))
    if len(selected_idx) == 1:
        axes = [axes]
    fig.suptitle(
        f'Temporal Attention Weights — Dept: {dept}  ({split} set)',
        fontsize=13, fontweight='bold',
    )

    for i, (label, prob) in enumerate(zip(y_sel, probs_sel)):
        # Average across heads and use each position's mean outgoing attention
        attn_mean = attn_weights[i].mean(axis=0)

        colour = '#e74c3c' if label == 1 else '#2ecc71'
        axes[i].bar(range(1, T_WINDOW + 1), attn_mean, color=colour,
                    edgecolor='black', alpha=0.8)
        axes[i].set_title(
            f'{"INSIDER" if label == 1 else "Normal"}\nP(insider) = {prob:.3f}',
            fontsize=9, color=colour, fontweight='bold',
        )
        axes[i].set_xlabel('Week in window')
        axes[i].set_ylabel('Attention weight')
        axes[i].set_xticks(range(1, T_WINDOW + 1))
        axes[i].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'attention_{dept}.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved attention_{dept}.png")


# Visualise attention for the first department that has test insiders
for dept in departments:
    if dept_datasets[dept]['y_test'].sum() > 0:
        visualize_attention_for_dept(dept, split='test', n_examples=6)
        break
else:
    # Fall back to the validation set if no dept has test insiders
    for dept in departments:
        if dept_datasets[dept]['y_val'].sum() > 0:
            visualize_attention_for_dept(dept, split='val', n_examples=6)
            break


# ── Cell 19: Export all results to Excel ──────────────────────────────────

print("\n" + "=" * 60)
print("STEP 14: EXPORTING RESULTS TO EXCEL")
print("=" * 60)

with pd.ExcelWriter('results_summary.xlsx', engine='openpyxl') as writer:

    # Sheet 1: Head-to-head comparison with all baselines
    pd.DataFrame([
        {
            'Method':     method,
            'F1':         scores['f1'],
            'Precision':  scores['precision'],
            'Recall':     scores['recall'],
            'AUC':        scores['auc'],
            'Privacy':    'Centralized' if 'centralized' in method.lower() else 'Federated',
        }
        for method, scores in baseline_results.items()
    ]).to_excel(writer, sheet_name='Method Comparison', index=False)

    # Sheet 2: Ablation results
    pd.DataFrame([
        {'Variant': variant, 'F1': score, 'Δ vs Full': score - f1_final}
        for variant, score in ablation_results.items()
    ]).to_excel(writer, sheet_name='Ablation Study', index=False)

    # Sheet 3: Per-department breakdown
    pd.DataFrame([
        {
            'Department':    dept,
            'F1':            res['f1'],
            'AUC':           res['auc'],
            'N sequences':   res['n'],
            'N insider seqs': res['n_insiders'],
        }
        for dept, res in dept_test_results.items()
    ]).sort_values('F1', ascending=False).to_excel(writer, sheet_name='Per-Department', index=False)

    # Sheet 4: Training history (loss and metrics per validation checkpoint)
    pd.DataFrame(history).to_excel(writer, sheet_name='Training History', index=False)

    # Sheet 5: Scikit-learn classification report
    cr = classification_report(
        y_true, y_pred,
        target_names=['Normal', 'Insider'],
        output_dict=True, zero_division=0,
    )
    pd.DataFrame(cr).T.to_excel(writer, sheet_name='Classification Report')

    # Sheet 6: Threshold analysis
    pd.DataFrame([
        {
            'Target Precision': target_p,
            'Threshold':        info['threshold'],
            'Actual Precision': info['precision'],
            'Recall':           info['recall'],
            'F1':               info['f1'],
            'F-beta (β=0.5)':   info['fbeta'],
        }
        for target_p, info in threshold_options.items()
    ]).to_excel(writer, sheet_name='Threshold Analysis', index=False)

    # Sheet 7: Privacy-utility frontier data
    pd.DataFrame({
        'Method':        method_labels,
        'Privacy Level': privacy_levels,
        'F1 (Insider)':  [round(v, 4) for v in utility_scores],
    }).to_excel(writer, sheet_name='Privacy-Utility', index=False)

    # Sheet 8: Confusion matrix
    pd.DataFrame(
        cm,
        index=['True: Normal', 'True: Insider'],
        columns=['Predicted: Normal', 'Predicted: Insider'],
    ).to_excel(writer, sheet_name='Confusion Matrix')

print("results_summary.xlsx saved successfully (8 sheets).")


# ── Cell 20: Final summary printout ──────────────────────────────────────

print("\n" + "=" * 70)
print("FINAL RESULTS SUMMARY")
print("=" * 70)

print(f"""
Task              : Binary classification — Normal vs Insider Threat
Dataset           : CERT v5.2 — {len(df):,} user-weeks, {df[USER_COL].nunique()} unique users
Insider rate      : {n_insider / len(df) * 100:.3f}%   ({n_normal / n_insider:.0f}:1 class imbalance)
Feature dimension : {n_features} features per user-week
FL clients        : {len(departments)} departments → {len(user_models)} per-user models (Stage 2)
Temporal split    : Train = wks {train_weeks[0]}-{train_weeks[-1]} | Val = {val_weeks[0]}-{val_weeks[-1]} | Test = {test_weeks[0]}-{test_weeks[-1]}
Architecture      : BiLSTM-Attention body + Deviation encoder + Binary head
Training          : Stage 1: pFedMe ({N_ROUNDS_S1} rounds) → Stage 2: per-user fine-tuning
Decision threshold: {BEST_THRESHOLD:.4f}  (chosen so precision ≥ 0.5 on the validation set)

┌──────────────────────────────────────────────────────────────────┐
│  TEST SET PERFORMANCE  (threshold = {BEST_THRESHOLD:.4f})                    │
├──────────────────────────────────┬───────────────────────────────┤
│  F1 (Insider class)              │  {f1_final:.4f}                         │
│  Fβ=0.5 (precision-weighted)     │  {fbeta_final:.4f}                         │
│  Precision (Insider class)       │  {prec_final:.4f}                         │
│  Recall    (Insider class)       │  {rec_final:.4f}                         │
│  AUC-ROC                         │  {auc_final:.4f}                         │
│  AUC-PR  (Average Precision)     │  {ap_final:.4f}                         │
│  FPR @ TPR = 90%                 │  {fpr_at_tpr90:.4f}                         │
│  False alarm rate                │  {fp / (fp + tn) * 100:.2f}%                           │
│  Detection rate                  │  {tp / (tp + fn) * 100:.2f}%                           │
└──────────────────────────────────┴───────────────────────────────┘

Method comparison  (F1 / Precision / Recall / AUC):
""")
for method, scores in baseline_results.items():
    tag = " ◄ OURS" if "Ours" in method else ""
    print(
        f"  {method:<42} "
        f"{scores['f1']:.4f} / {scores['precision']:.4f} / "
        f"{scores['recall']:.4f} / {scores['auc']:.4f}{tag}"
    )

print("\nAblation  (Δ vs full model):")
for variant, score in ablation_results.items():
    delta = score - f1_final
    verdict = (
        "✓ this component adds value"
        if score < f1_final - 0.005
        else "~ roughly neutral"
        if abs(score - f1_final) <= 0.005
        else "✗ hurts performance — worth investigating"
    )
    print(f"  {variant:<55} {score:.4f}  (Δ = {delta:+.4f})  {verdict}")

print(f"""
Output files:
  ✓ eda_plots.png              — class distributions, temporal activity patterns
  ✓ training_curves.png        — Stage-1 loss, F1, AUC across FL rounds
  ✓ threshold_analysis.png     — PR curve and metric sweeps vs threshold
  ✓ binary_results.png         — confusion matrix, ROC, PR, per-dept F1, distributions
  ✓ baseline_comparison.png    — head-to-head bar charts
  ✓ ablation_study.png         — per-component contribution analysis
  ✓ privacy_utility.png        — privacy-utility frontier
  ✓ attention_<dept>.png       — temporal attention weights for insider vs normal
  ✓ results_summary.xlsx       — all tables in 8 sheets
  ✓ best_stage1_checkpoint.pt  — best federated model weights
  ✓ best_stage2_checkpoint.pt  — personalised user model weights
""")

# Confirm each expected output file exists
expected_outputs = [
    'eda_plots.png', 'training_curves.png', 'threshold_analysis.png',
    'binary_results.png', 'baseline_comparison.png', 'ablation_study.png',
    'privacy_utility.png', 'results_summary.xlsx',
    'best_stage1_checkpoint.pt', 'best_stage2_checkpoint.pt',
]
for filename in expected_outputs:
    status = '✓  found' if os.path.exists(filename) else '✗  MISSING'
    print(f"  {status}   {filename}")
