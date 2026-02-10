"""
IMPROVED STGNN SLEEP STAGING TRAINING v2 - UPDATED FOR RESEARCH-GRADE PREPROCESSING
Key improvements:
1. Compatible with 6-channel input (EEG, EOG, EMG, Resp, Temp)
2. Compatible with 5 canonical frequency bands (non-overlapping)
3. Class-specific precision losses with stronger penalties for N3/REM over-prediction
4. Threshold calibration for minority classes
5. Improved N1/N2 boundary handling
6. Enhanced feature discrimination
"""

import os
import math
import time
import copy
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report, cohen_kappa_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ----------------------------- CONFIG -----------------------------
DATA_NPZ = "preprocessed_sleep_edf_research_final.npz"  # Updated for new preprocessing
MODEL_OUT = "best_stgnn_v2_research_improved.pt"
TRAIN_CURVES = "training_curves_stgnn_v2_research.png"
CONF_MAT = "confusion_matrix_stgnn_v2_research.png"
EDGE_VIZ_DIR = "edges_visualizations_v2_research"

SEED = 42
BATCH_SIZE = 64
GRAD_ACCUM_STEPS = 2
EPOCHS = 150
LEARNING_RATE = 2e-4
WARMUP_EPOCHS = 12

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import platform
IS_WINDOWS = platform.system() == 'Windows'
NUM_WORKERS = 0 if IS_WINDOWS else (4 if torch.cuda.is_available() else 0)
PIN_MEMORY = False
PERSISTENT_WORKERS = False

NUM_CLASSES = 5
LAMBDA_S = 2e-3
LAMBDA_T = 2e-3
LAMBDA_GP = 2e-5
LAMBDA_CONFUSION = 0.25
LAMBDA_N2_FOCUS = 0.30
LAMBDA_N3_FOCUS = 0.45  # Increased
LAMBDA_PRECISION = 0.35  # Significantly increased
LAMBDA_N3_PRECISION = 0.50  # New: specific for N3
LAMBDA_REM_PRECISION = 0.30  # New: specific for REM
LAMBDA_BOUNDARY = 0.20  # New: N1/N2 boundary loss
CHEB_K = 3

HIDDEN_DIM = 256  # Increased capacity

LABEL_SMOOTHING = 0.015
MIXUP_ALPHA = 0.15

DROPOUT_PATH_RATE = 0.15
GLOBAL_WEIGHT_DECAY = 1.5e-3
NOISE_STD_DE = 0.012

RAW_SEQ_LEN = 150
RAW_FEAT = 64

OVERFIT_THRESHOLD = 0.12
OVERFIT_PATIENCE = 5
LR_REDUCE_FACTOR = 0.5
LR_REDUCE_MIN = 1e-7

# Class-specific thresholds (will be learned)
CLASS_THRESHOLDS = {
    1: 0.35,  # N1: higher threshold to reduce false positives
    3: 0.55,  # N3: much higher threshold to reduce over-prediction
    4: 0.40,  # REM: higher threshold
}

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

# ------------------------- Utilities -------------------------
def row_softmax(mat, eps=1e-8):
    mat = mat - mat.max(dim=-1, keepdim=True)[0]
    expm = torch.exp(mat)
    denom = expm.sum(dim=-1, keepdim=True) + eps
    return expm / denom

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

# ------------------------- Dataset & Sampler -------------------------
class WindowedDEData(Dataset):
    def __init__(self, X, y, subject_ids=None):
        self.X = X
        self.y = y
        self.subject_ids = subject_ids if subject_ids is not None else np.arange(len(y))
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], int(self.y[idx]), int(self.subject_ids[idx])

def get_improved_sampler(y_train, target_distribution=None):
    counts = np.bincount(y_train, minlength=NUM_CLASSES)
    if target_distribution is not None:
        desired = np.array([target_distribution.get(i, 0.0) for i in range(NUM_CLASSES)], dtype=float)
        if desired.sum() <= 0:
            desired = np.ones(NUM_CLASSES, dtype=float) / NUM_CLASSES
        else:
            desired = desired / desired.sum()
        emp = np.maximum(counts / counts.sum(), 1e-9)
        weights_per_class = desired / emp
        weights_per_class = weights_per_class / weights_per_class.sum() * NUM_CLASSES
        sample_weights = weights_per_class[y_train].astype(np.float32)
    else:
        max_count = counts.max()
        with np.errstate(divide='ignore'):
            weights = np.sqrt(max_count / np.maximum(counts, 1))
        weights = weights / weights.sum() * NUM_CLASSES
        sample_weights = weights[y_train].astype(np.float32)
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

# --------------------- Model components ---------------------
class LearnAdjacency(nn.Module):
    def __init__(self, feature_dim, bias=True):
        super().__init__()
        self.w = nn.Parameter(torch.randn(feature_dim) * 0.1)
        if bias:
            self.b = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter('b', None)
        self.layer_norm = nn.LayerNorm(1)
    def forward(self, X):
        batch, N, feat_dim = X.shape
        diff = torch.abs(X.unsqueeze(2) - X.unsqueeze(1))
        logits = torch.einsum('bnjf,f->bnj', diff, self.w)
        if self.b is not None:
            logits = logits + self.b
        logits = F.relu(logits)
        logits = self.layer_norm(logits.unsqueeze(-1)).squeeze(-1)
        adj = row_softmax(logits)
        return adj

class ChebConv(nn.Module):
    def __init__(self, in_channels, out_channels, K=3, bias=True):
        super().__init__()
        self.K = K
        self.in_c = in_channels
        self.out_c = out_channels
        self.weights = nn.Parameter(torch.randn(K, in_channels, out_channels) * (1.0 / math.sqrt(in_channels)))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        self.layer_norm = nn.LayerNorm(out_channels)
    def forward(self, x, adj):
        batch, N, Fin = x.shape
        device = x.device
        I = torch.eye(N, device=device).unsqueeze(0).expand(batch, -1, -1)
        A_hat = adj + I
        row_sum = A_hat.sum(-1, keepdim=True).clamp(min=1e-6)
        P = A_hat / row_sum
        out = torch.zeros(batch, N, self.weights.shape[2], device=device)
        Pk_x = x
        for k in range(self.K):
            if k == 0:
                Pk_x = x
            else:
                Pk_x = torch.bmm(P, Pk_x)
            Wk = self.weights[k]
            out = out + torch.einsum('bnf,fo->bno', Pk_x, Wk)
        if self.bias is not None:
            out = out + self.bias
        out = self.layer_norm(out)
        return out

class SpatialGraphModule(nn.Module):
    def __init__(self, in_feats, node_hidden=128, cheb_K=3, num_channels=6, drop_path=0.15):
        """
        Updated for 6 channels: EEG Fpz-Cz, EEG Pz-Oz, EOG, EMG, Resp, Temp
        """
        super().__init__()
        self.in_feats = in_feats
        self.node_hidden = node_hidden
        self.num_channels = num_channels
        self.fe_proj = nn.Sequential(
            nn.Linear(in_feats, node_hidden),
            nn.LayerNorm(node_hidden),
            nn.GELU(),
            nn.Dropout(0.20)
        )
        self.adj_learner = LearnAdjacency(node_hidden)
        self.cheb = ChebConv(node_hidden, node_hidden, K=cheb_K)
        self.satt = nn.Sequential(
            nn.Linear(node_hidden * 2, node_hidden),
            nn.LayerNorm(node_hidden),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(node_hidden, 1)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
    def forward(self, X_de):
        if not torch.is_tensor(X_de):
            X_de = torch.from_numpy(X_de).float().to(next(self.parameters()).device)
        b, C, T, B = X_de.shape
        X_resh = X_de.permute(0, 2, 1, 3).contiguous().view(b * T, C, B)
        Xc_flat = X_resh.view(b * T * C, B)
        Xn_flat = self.fe_proj(Xc_flat)
        Xn = Xn_flat.view(b, T, C, self.node_hidden)
        SG_seq = []
        g_out_seq = []
        epoch_repr_list = []
        for t in range(T):
            Xn_t = Xn[:, t, :, :]
            SG_t = self.adj_learner(Xn_t)
            g_out_t = self.cheb(Xn_t, SG_t)
            g_out_t = self.drop_path(g_out_t)
            neighbor_mean_t = torch.bmm(SG_t, g_out_t)
            cat_t = torch.cat([g_out_t, neighbor_mean_t], dim=-1)
            att_logits_t = self.satt(cat_t).squeeze(-1)
            att_t = torch.sigmoid(att_logits_t).unsqueeze(-1)
            gated_t = g_out_t * att_t
            epoch_repr_t = gated_t.mean(dim=1)
            SG_seq.append(SG_t)
            g_out_seq.append(g_out_t)
            epoch_repr_list.append(epoch_repr_t)
        SG_seq = torch.stack(SG_seq, dim=1)
        g_out_seq = torch.stack(g_out_seq, dim=1)
        epoch_seq = torch.stack(epoch_repr_list, dim=1)
        return epoch_seq, SG_seq, Xn, g_out_seq

class TemporalGraphModule(nn.Module):
    def __init__(self, node_dim, cheb_K=2, lstm_hidden=128, drop_path=0.15):
        super().__init__()
        self.adj_learner = LearnAdjacency(node_dim)
        self.cheb = ChebConv(node_dim, node_dim, K=cheb_K)
        self.bilstm = nn.LSTM(node_dim, lstm_hidden, num_layers=2, batch_first=True, 
                              bidirectional=True, dropout=0.30)
        self.layer_norm = nn.LayerNorm(2 * lstm_hidden)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
    def forward(self, epoch_nodes):
        TG = self.adj_learner(epoch_nodes)
        t_out = self.cheb(epoch_nodes, TG)
        t_out = self.drop_path(t_out)
        lstm_out, _ = self.bilstm(t_out)
        lstm_out = self.layer_norm(lstm_out)
        agg = lstm_out.mean(dim=1)
        return agg, TG, lstm_out

class MDFMGatedFusion(nn.Module):
    def __init__(self, D_sp, D_temp, D_raw, out_dim):
        super().__init__()
        self.sp_proj = nn.Linear(D_sp, out_dim)
        self.temp_proj = nn.Linear(D_temp, out_dim)
        self.raw_proj = nn.Linear(D_raw, out_dim)
        self.gate = nn.Sequential(
            nn.Linear(out_dim * 3, out_dim * 2),
            nn.LayerNorm(out_dim * 2),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(out_dim * 2, out_dim),
            nn.Sigmoid()
        )
        self.out_proj = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(0.30)
        )
    def forward(self, sp, temp, raw):
        sp_p = self.sp_proj(sp)
        temp_p = self.temp_proj(temp)
        raw_p = self.raw_proj(raw)
        concat = torch.cat([sp_p, temp_p, raw_p], dim=-1)
        g = self.gate(concat)
        fused = g * sp_p + (1 - g) * temp_p + raw_p * 0.15
        return self.out_proj(fused)

class STGNNSleepModel(nn.Module):
    def __init__(self, num_channels=6, f_de=5, T=9, raw_feat=64, hidden_dim=256, cheb_K=3):
        """
        Updated model for research-grade preprocessing:
        - num_channels=6: EEG Fpz-Cz, EEG Pz-Oz, EOG, EMG, Resp, Temp
        - f_de=5: canonical non-overlapping frequency bands
        - T=9: temporal receptive field (context epochs)
        """
        super().__init__()
        self.num_channels = num_channels
        self.T = T
        self.f_de = f_de
        self.spatial_module = SpatialGraphModule(in_feats=f_de, node_hidden=128, cheb_K=cheb_K, 
                                                 num_channels=num_channels, drop_path=DROPOUT_PATH_RATE)
        self.temporal_module = TemporalGraphModule(node_dim=128, cheb_K=2, lstm_hidden=128, 
                                                    drop_path=DROPOUT_PATH_RATE)
        self.raw_lstm = nn.LSTM(raw_feat, hidden_dim//2, num_layers=2, batch_first=True, 
                                bidirectional=True, dropout=0.30)
        self.raw_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True, dropout=0.18)
        self.fusion = MDFMGatedFusion(D_sp=128, D_temp=2*128, D_raw=hidden_dim, out_dim=hidden_dim)
        
        # Class-specific discriminators for better boundary detection
        self.class_branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim//2),
                nn.LayerNorm(hidden_dim//2),
                nn.GELU(),
                nn.Dropout(0.30),
                nn.Linear(hidden_dim//2, 1)
            ) for _ in range(NUM_CLASSES)
        ])
        
        self.final = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(hidden_dim, NUM_CLASSES)
        )
        
    def forward(self, x_de, x_raw):
        epoch_seq, SG_seq, Xn_seq, g_out_seq = self.spatial_module(x_de)
        temp_agg, TG, lstm_seq = self.temporal_module(epoch_seq)
        sp_agg = epoch_seq.mean(dim=1)
        raw_out, _ = self.raw_lstm(x_raw)
        q = raw_out.mean(dim=1, keepdim=True)
        att_out, _ = self.raw_attention(q, raw_out, raw_out)
        raw_features = att_out.squeeze(1)
        fused = self.fusion(sp_agg, temp_agg, raw_features)
        main_out = self.final(fused)
        class_outs = torch.cat([branch(fused) for branch in self.class_branches], dim=1)
        return 0.65 * main_out + 0.35 * class_outs, SG_seq, TG, Xn_seq, g_out_seq, epoch_seq

# ---------------------- Enhanced Losses ----------------------
class AdaptiveFocalLoss(nn.Module):
    def __init__(self, alpha=None, base_gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.base_gamma = base_gamma
        self.label_smoothing = label_smoothing
        self.class_gammas = nn.Parameter(
            torch.tensor([1.8, 2.8, 1.8, 1.6, 2.2], dtype=torch.float32), 
            requires_grad=False
        )
    
    def update_gammas(self, class_f1_scores):
        with torch.no_grad():
            # N1 (class 1): boost gamma if F1 low
            if class_f1_scores[1] < 0.40:
                self.class_gammas[1] = min(3.5, self.class_gammas[1] + 0.15)
            elif class_f1_scores[1] > 0.50:
                self.class_gammas[1] = max(2.2, self.class_gammas[1] - 0.08)
            
            # N3 (class 3): increase if F1 low
            if class_f1_scores[3] < 0.50:
                self.class_gammas[3] = min(2.5, self.class_gammas[3] + 0.18)
            elif class_f1_scores[3] > 0.70:
                self.class_gammas[3] = max(1.2, self.class_gammas[3] - 0.10)
            
            # Other classes
            for i in [0, 2, 4]:
                if class_f1_scores[i] < 0.85:
                    self.class_gammas[i] = min(2.8, self.class_gammas[i] + 0.10)
                elif class_f1_scores[i] > 0.92:
                    self.class_gammas[i] = max(1.4, self.class_gammas[i] - 0.08)
    
    def forward(self, inputs, targets):
        if self.label_smoothing > 0:
            n_class = inputs.size(1)
            one_hot = torch.zeros_like(inputs).scatter(1, targets.unsqueeze(1), 1)
            one_hot = one_hot * (1 - self.label_smoothing) + self.label_smoothing / n_class
            log_prb = F.log_softmax(inputs, dim=1)
            ce_loss = -(one_hot * log_prb).sum(dim=1)
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = torch.zeros_like(ce_loss)
        for i in range(NUM_CLASSES):
            mask = (targets == i)
            if mask.any():
                gamma = self.class_gammas[i].item()
                focal_loss[mask] = ((1 - pt[mask]) ** gamma) * ce_loss[mask]
        return focal_loss.mean()

class N3PrecisionLoss(nn.Module):
    """Heavily penalize false N3 predictions"""
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
    
    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        loss = torch.tensor(0.0, device=logits.device)
        
        # Penalize predicting N3 when it's not N3
        not_n3_mask = (targets != 3)
        if not_n3_mask.any():
            false_n3_prob = probs[not_n3_mask, 3]
            # Very strong penalty - cubic + margin
            loss = loss + (false_n3_prob ** 3.0).mean() * 5.0
            # Additional margin-based penalty
            margin_violation = torch.clamp(false_n3_prob - 0.15, min=0)
            loss = loss + (margin_violation ** 2).mean() * 3.0
        
        # Also ensure true N3 has high confidence
        n3_mask = (targets == 3)
        if n3_mask.any():
            n3_prob = probs[n3_mask, 3]
            loss = loss + -torch.log(n3_prob + 1e-7).mean() * 1.0
        
        return self.weight * loss

class REMPrecisionLoss(nn.Module):
    """Reduce false REM predictions"""
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
    
    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        loss = torch.tensor(0.0, device=logits.device)
        
        # Penalize predicting REM when it's not REM
        not_rem_mask = (targets != 4)
        if not_rem_mask.any():
            false_rem_prob = probs[not_rem_mask, 4]
            loss = loss + (false_rem_prob ** 2.8).mean() * 3.5
            # Margin penalty
            margin_violation = torch.clamp(false_rem_prob - 0.20, min=0)
            loss = loss + (margin_violation ** 2).mean() * 2.0
        
        return self.weight * loss

class BoundaryLoss(nn.Module):
    """Help discriminate between N1 and N2"""
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
    
    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        loss = torch.tensor(0.0, device=logits.device)
        
        # For N1 samples, penalize N2 probability
        n1_mask = (targets == 1)
        if n1_mask.any():
            n2_prob_when_n1 = probs[n1_mask, 2]
            loss = loss + (n2_prob_when_n1 ** 2.2).mean() * 2.5
        
        # For N2 samples, penalize N1 probability
        n2_mask = (targets == 2)
        if n2_mask.any():
            n1_prob_when_n2 = probs[n2_mask, 1]
            loss = loss + (n1_prob_when_n2 ** 2.2).mean() * 2.0
        
        return self.weight * loss

class N3FocusLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        loss = torch.tensor(0.0, device=logits.device)
        n3_mask = (targets == 3)
        if n3_mask.any():
            n3_prob = probs[n3_mask, 3]
            n2_prob = probs[n3_mask, 2]
            loss = loss + -torch.log(n3_prob + 1e-7).mean() * 0.8
            loss = loss + (n2_prob ** 2.2).mean() * 3.5
        n2_mask = (targets == 2)
        if n2_mask.any():
            false_n3_prob = probs[n2_mask, 3]
            loss = loss + (false_n3_prob ** 2.5).mean() * 3.0
        return self.weight * loss

class N2FocusLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        loss = torch.tensor(0.0, device=logits.device)
        n2_mask = (targets == 2)
        if n2_mask.any():
            n2_prob = probs[n2_mask, 2]
            n3_prob = probs[n2_mask, 3]
            n1_prob = probs[n2_mask, 1]
            loss = loss + -torch.log(n2_prob + 1e-7).mean() * 0.6
            loss = loss + (n3_prob ** 2.0).mean() * 2.5
            loss = loss + (n1_prob ** 2.0).mean() * 1.5
        return self.weight * loss

class PrecisionLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        loss = torch.tensor(0.0, device=logits.device)
        not_n1_mask = (targets != 1)
        if not_n1_mask.any():
            false_n1_prob = probs[not_n1_mask, 1]
            loss = loss + (false_n1_prob ** 2.8).mean() * 2.5
        return self.weight * loss

class ConfusionPenalty(nn.Module):
    def __init__(self, confusion_pairs, penalty_weight=1.0):
        super().__init__()
        self.confusion_pairs = confusion_pairs
        self.penalty_weight = penalty_weight
    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        penalty = torch.tensor(0.0, device=logits.device)
        for true_class, confused_class, weight in self.confusion_pairs:
            mask = (targets == true_class)
            if mask.any():
                confused_prob = probs[mask, confused_class]
                penalty = penalty + weight * (confused_prob ** 2.8).mean()
        return self.penalty_weight * penalty

def spatial_graph_loss_seq(SG_seq, X_nodes_seq, lambda_frob=1e-3, lambda_smooth=1e-3):
    b, T, N, N2 = SG_seq.shape
    loss_total = 0.0
    for t in range(T):
        SG_t = SG_seq[:, t, :, :]
        Xn_t = X_nodes_seq[:, t, :, :]
        frob = ((SG_t - SG_t.transpose(1,2))**2).sum()
        diff = (Xn_t.unsqueeze(2) - Xn_t.unsqueeze(1))**2
        dist = diff.sum(-1)
        sim_term = (SG_t * dist).sum()
        sparsity = (SG_t ** 2).sum()
        loss_total = loss_total + (lambda_frob * frob + lambda_smooth * sim_term + 1e-4 * sparsity)
    return loss_total / float(T)

def temporal_graph_loss(TG, epochs_repr, lambda_frob=1e-3, lambda_smooth=1e-3):
    frob = ((TG - TG.transpose(1,2))**2).sum()
    diff = (epochs_repr.unsqueeze(2) - epochs_repr.unsqueeze(1))**2
    dist = diff.sum(-1)
    sim_term = (TG * dist).sum()
    sparsity = (TG ** 2).sum()
    return lambda_frob * frob + lambda_smooth * sim_term + 1e-4 * sparsity

# ---------------------- Mixup & Calibration ----------------------
def mixup_data(x_de, x_raw, y, alpha=0.4):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1 - lam)
    else:
        lam = 1
    batch_size = x_de.size(0)
    index = torch.randperm(batch_size).to(x_de.device)
    mixed_x_de = lam * x_de + (1 - lam) * x_de[index, :]
    mixed_x_raw = lam * x_raw + (1 - lam) * x_raw[index, :]
    y_a, y_b = y, y[index]
    return mixed_x_de, mixed_x_raw, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def apply_calibrated_thresholds(logits, thresholds):
    """Apply class-specific thresholds for better precision"""
    probs = F.softmax(logits, dim=1)
    adjusted_probs = probs.clone()
    
    for cls, threshold in thresholds.items():
        # Reduce probability if below threshold
        mask = probs[:, cls] < threshold
        adjusted_probs[mask, cls] = adjusted_probs[mask, cls] * 0.5
    
    return adjusted_probs.argmax(dim=1)

# ---------------------- Collate & RawProjector ----------------------
def collate_fn_factory(device):
    def collate_fn(batch):
        Xs = np.stack([item[0] for item in batch], axis=0)
        ys = np.array([item[1] for item in batch], dtype=np.int64)
        sids = np.array([item[2] for item in batch], dtype=np.int64)
        Xs_t = torch.from_numpy(Xs).float().to(device)
        ys_t = torch.from_numpy(ys).long().to(device)
        sids_t = torch.from_numpy(sids).long().to(device)
        return Xs_t, ys_t, sids_t
    return collate_fn

class RawProjector(nn.Module):
    def __init__(self, f_de, raw_feat=RAW_FEAT, raw_seq_len=RAW_SEQ_LEN):
        super().__init__()
        self.f_de = f_de
        self.raw_feat = raw_feat
        self.raw_seq_len = raw_seq_len
        self.step_proj = nn.Linear(f_de, raw_feat)
        self.temporal_smooth = nn.Sequential(
            nn.Conv1d(raw_feat, raw_feat, kernel_size=3, padding=1),
            nn.GELU()
        )
    def forward(self, X_de):
        de_seq = X_de.mean(dim=1)
        B, T, F = de_seq.shape
        x = de_seq.view(B * T, F)
        x = self.step_proj(x)
        x = x.view(B, T, self.raw_feat)
        factor = math.ceil(self.raw_seq_len / float(T))
        x_rep = x.repeat_interleave(factor, dim=1)
        if x_rep.shape[1] >= self.raw_seq_len:
            x_rep = x_rep[:, :self.raw_seq_len, :]
        else:
            pad_len = self.raw_seq_len - x_rep.shape[1]
            pad = x_rep[:, -1:, :].repeat(1, pad_len, 1)
            x_rep = torch.cat([x_rep, pad], dim=1)
        x_rep = x_rep.permute(0, 2, 1)
        x_rep = self.temporal_smooth(x_rep)
        x_rep = x_rep.permute(0, 2, 1)
        return x_rep

# ---------------------- Monitor & plotting ----------------------
class ImprovedTrainingMonitor:
    def __init__(self, patience=25, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = float('inf')
        self.best_val_f1 = 0.0
        self.best_balanced_metric = 0.0
        self.wait = 0
        self.best_epoch = 0
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_f1': [], 'val_f1': [], 'lr': [], 
            'confusion_loss': [], 'n3_precision_loss': []
        }
    def update(self, train_loss, val_loss, train_f1, val_f1, epoch, lr, conf_loss=0.0, n3_prec_loss=0.0):
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_f1'].append(train_f1)
        self.history['val_f1'].append(val_f1)
        self.history['lr'].append(lr)
        self.history['confusion_loss'].append(conf_loss)
        self.history['n3_precision_loss'].append(n3_prec_loss)
        balanced_metric = val_f1 - 0.08 * val_loss
        improved = False
        if balanced_metric > self.best_balanced_metric + self.min_delta:
            self.best_balanced_metric = balanced_metric
            self.best_val_f1 = val_f1
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self.wait = 0
            improved = True
        else:
            self.wait += 1
        return improved, (self.wait >= self.patience)

def plot_training_curves(monitor, save_path=TRAIN_CURVES):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    epochs = range(1, len(monitor.history['train_loss']) + 1)
    axes[0, 0].plot(epochs, monitor.history['train_loss'], label='Train Loss', alpha=0.8)
    axes[0, 0].plot(epochs, monitor.history['val_loss'], label='Val Loss', alpha=0.8)
    axes[0, 0].axvline(monitor.best_epoch, color='r', linestyle='--', alpha=0.5, label='Best')
    axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss'); axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)
    axes[0, 0].set_title('Loss Curves')
    axes[0, 1].plot(epochs, monitor.history['train_f1'], label='Train F1', alpha=0.8)
    axes[0, 1].plot(epochs, monitor.history['val_f1'], label='Val F1', alpha=0.8)
    axes[0, 1].axvline(monitor.best_epoch, color='r', linestyle='--', alpha=0.5, label='Best')
    axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('F1 Score'); axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)
    axes[0, 1].set_title('F1 Score Curves')
    axes[1, 0].plot(epochs, monitor.history['lr'], alpha=0.8); axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('Learning Rate'); axes[1, 0].grid(alpha=0.3); axes[1, 0].set_title('LR'); axes[1, 0].set_yscale('log')
    gap = np.array(monitor.history['train_f1']) - np.array(monitor.history['val_f1'])
    axes[1, 1].plot(epochs, gap, alpha=0.8, color='purple'); axes[1, 1].axhline(0, color='black', linestyle='--', alpha=0.3)
    axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('Train F1 - Val F1'); axes[1, 1].grid(alpha=0.3); axes[1, 1].set_title('Overfitting Gap')
    plt.tight_layout(); plt.savefig(save_path, dpi=200, bbox_inches='tight'); plt.close()

def save_adj_heatmap(mat, title, fname, xticklabels=None, yticklabels=None, vmin=None, vmax=None):
    plt.figure(figsize=(10, 8))  # Larger size for 6×6 adjacency matrices
    sns.heatmap(mat, annot=True, fmt=".2f", xticklabels=xticklabels, yticklabels=yticklabels, vmin=vmin, vmax=vmax, cmap='viridis')
    plt.title(title); plt.tight_layout(); plt.savefig(fname, dpi=200); plt.close()

# ---------------------- Training Pipeline ----------------------
def parse_target_distribution(raw_td):
    if raw_td is None:
        return None
    if isinstance(raw_td, dict):
        return {int(k): float(v) for k, v in raw_td.items()}
    if isinstance(raw_td, np.ndarray):
        if raw_td.dtype == object:
            try:
                lst = raw_td.tolist()
                if isinstance(lst, dict):
                    return {int(k): float(v) for k, v in lst.items()}
                if isinstance(lst, list):
                    try:
                        return {int(k): float(v) for k, v in lst}
                    except Exception:
                        pass
            except Exception:
                pass
        if raw_td.ndim == 1 and raw_td.size == NUM_CLASSES:
            return {i: float(raw_td[i]) for i in range(NUM_CLASSES)}
        if raw_td.ndim == 2 and raw_td.shape[1] >= 2:
            try:
                return {int(int_row[0]): float(int_row[1]) for int_row in raw_td}
            except Exception:
                pass
    try:
        lst = list(raw_td)
        if len(lst) == NUM_CLASSES:
            return {i: float(lst[i]) for i in range(NUM_CLASSES)}
    except Exception:
        pass
    return None

def train_pipeline(npz_path=DATA_NPZ):
    print(f"[{datetime.now()}] Loading data from {npz_path}")
    data = np.load(npz_path, allow_pickle=True)

    if 'X' in data:
        X = data['X']
    elif 'x' in data:
        X = data['x']
    else:
        raise KeyError("NPZ must contain key 'X'")

    y = data['y']
    subject_ids = data.get('subject_ids', np.arange(len(y)))
    channel_names = data.get('global_channel_names', data.get('channel_names', None))
    raw_td = data.get('target_distribution', None)
    target_distribution = parse_target_distribution(raw_td)
    if raw_td is not None and target_distribution is None:
        print("Warning: couldn't parse target_distribution from NPZ; falling back to sampler default.")
    f_de = int(data.get('f_de', X.shape[-1]))

    print(f"Loaded: X={X.shape}, y={y.shape}")
    N, C, T, Bn = X.shape
    print(f"Channels: {C}, Temporal receptive field (T): {T}, DE frequency bands: {Bn}, f_de stored: {f_de}")
    
    # Validate 6-channel configuration (research-grade preprocessing)
    if C != 6:
        raise ValueError(
            f"Expected 6 channels (EEG Fpz-Cz, EEG Pz-Oz, EOG, EMG, Resp, Temp), got {C}. "
            f"Please ensure your preprocessing includes all 6 signals."
        )
    print(f"✓ Full 6-channel configuration confirmed (EEG×2, EOG, EMG, Resp, Temp)")
    
    # Validate 5 canonical frequency bands
    if Bn != 5:
        raise ValueError(
            f"Expected 5 canonical frequency bands (Delta, Theta, Alpha, Beta, Gamma), got {Bn}. "
            f"Please check your preprocessing configuration."
        )
    print(f"✓ Canonical 5 frequency bands confirmed")

    if channel_names is not None:
        print(f"Channel names: {list(channel_names)}")
        expected_channels = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'EMG submental', 'Resp oro-nasal', 'Temp rectal']
        if not all(ch in channel_names for ch in ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'EMG submental']):
            print(f"⚠ Warning: Standard channel names not found. Expected channels include: {expected_channels}")

    unique_subs = np.unique(subject_ids)
    np.random.seed(SEED)
    np.random.shuffle(unique_subs)
    n_train = int(0.7 * len(unique_subs))
    n_val = int(0.15 * len(unique_subs))
    train_subs = unique_subs[:n_train]
    val_subs = unique_subs[n_train:n_train+n_val]
    test_subs = unique_subs[n_train+n_val:]

    train_mask = np.isin(subject_ids, train_subs)
    val_mask = np.isin(subject_ids, val_subs)
    test_mask = np.isin(subject_ids, test_subs)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    subs_train = subject_ids[train_mask]

    print(f"Split: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")
    print(f"Train class distribution: {np.bincount(y_train, minlength=NUM_CLASSES)}")
    print(f"Val class distribution: {np.bincount(y_val, minlength=NUM_CLASSES)}")

    collate_fn = collate_fn_factory(DEVICE)

    train_dataset = WindowedDEData(X_train, y_train, subs_train)
    val_dataset = WindowedDEData(X_val, y_val, subject_ids[val_mask])
    test_dataset = WindowedDEData(X_test, y_test, subject_ids[test_mask])

    train_sampler = get_improved_sampler(y_train, target_distribution=target_distribution)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS)

    model = STGNNSleepModel(num_channels=6, f_de=5, T=T, raw_feat=RAW_FEAT, hidden_dim=HIDDEN_DIM, cheb_K=CHEB_K).to(DEVICE)
    raw_projector = RawProjector(f_de=5, raw_feat=RAW_FEAT, raw_seq_len=RAW_SEQ_LEN).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in raw_projector.parameters())
    trainable_params = sum(p.numel() for p in list(model.parameters()) + list(raw_projector.parameters()) if p.requires_grad)
    print(f"Total params (model+projector): {total_params:,}, Trainable: {trainable_params:,}")

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    cw_full = np.ones(NUM_CLASSES, dtype=np.float32)
    cw_full[:len(class_weights)] = class_weights.astype(np.float32)
    # Boost minority classes more
    cw_full[3] *= 2.00  # N3
    cw_full[1] *= 1.80  # N1
    cw_full[2] *= 1.35  # N2
    cw_full[4] *= 1.25  # REM
    cw_full[0] *= 0.85  # Wake
    class_weights_t = torch.tensor(cw_full, dtype=torch.float32).to(DEVICE)

    criterion = AdaptiveFocalLoss(alpha=class_weights_t, base_gamma=2.0, label_smoothing=LABEL_SMOOTHING)
    precision_loss = PrecisionLoss(weight=LAMBDA_PRECISION).to(DEVICE)
    n2_focus_loss = N2FocusLoss(weight=LAMBDA_N2_FOCUS).to(DEVICE)
    n3_focus_loss = N3FocusLoss(weight=LAMBDA_N3_FOCUS).to(DEVICE)
    n3_precision_loss = N3PrecisionLoss(weight=LAMBDA_N3_PRECISION).to(DEVICE)
    rem_precision_loss = REMPrecisionLoss(weight=LAMBDA_REM_PRECISION).to(DEVICE)
    boundary_loss = BoundaryLoss(weight=LAMBDA_BOUNDARY).to(DEVICE)

    confusion_pairs = [
        (3, 2, 4.0), (2, 3, 3.5),  # N3/N2 confusion - increased
        (2, 1, 2.8), (1, 2, 2.5),  # N2/N1 confusion - increased
        (2, 4, 1.5), (4, 2, 1.5),  # N2/REM
        (1, 4, 1.2), (4, 1, 1.2),  # N1/REM
        (1, 3, 0.8), (3, 1, 0.8)   # N1/N3
    ]
    confusion_penalty = ConfusionPenalty(confusion_pairs, penalty_weight=LAMBDA_CONFUSION).to(DEVICE)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(raw_projector.parameters()), 
        lr=LEARNING_RATE, 
        weight_decay=GLOBAL_WEIGHT_DECAY, 
        betas=(0.9, 0.999)
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-7)
    reduce_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=LR_REDUCE_FACTOR, patience=4, min_lr=LR_REDUCE_MIN)
    def warmup_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        return 1.0
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)

    scaler = GradScaler()
    monitor = ImprovedTrainingMonitor(patience=28, min_delta=0.001)

    best_state = None
    best_meta = {}
    overfit_window = deque(maxlen=OVERFIT_PATIENCE)

    print(f"\nStarting training on {DEVICE}")
    print(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUM_STEPS}")
    print(f"Class weights: {cw_full}")
    if target_distribution is not None:
        print(f"Using target_distribution in sampler: {target_distribution}")

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        raw_projector.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []

        confusion_loss_accum = 0.0
        n3_prec_loss_accum = 0.0

        optimizer.zero_grad()

        for batch_idx, (Xb, yb, sids) in enumerate(train_loader):
            if NOISE_STD_DE > 0:
                noise = torch.randn_like(Xb) * NOISE_STD_DE
                Xb_aug = Xb + noise
            else:
                Xb_aug = Xb

            rawb = raw_projector(Xb_aug)

            # More selective mixup - focus on N1 and N3
            use_mixup = (np.random.rand() < 0.20) and torch.any((yb == 1) | (yb == 3))
            if use_mixup:
                Xb_used, rawb_used, y_a, y_b, lam = mixup_data(Xb_aug, rawb, yb, alpha=MIXUP_ALPHA)
            else:
                Xb_used, rawb_used = Xb_aug, rawb

            with autocast():
                outputs, SG_seq, TG, Xn_seq, g_out_seq, epoch_seq = model(Xb_used, rawb_used)

                if use_mixup:
                    loss_ce = mixup_criterion(criterion, outputs, y_a, y_b, lam)
                    loss_conf = torch.tensor(0.0, device=DEVICE)
                    loss_n2 = torch.tensor(0.0, device=DEVICE)
                    loss_n3 = torch.tensor(0.0, device=DEVICE)
                    loss_prec = torch.tensor(0.0, device=DEVICE)
                    loss_n3_prec = torch.tensor(0.0, device=DEVICE)
                    loss_rem_prec = torch.tensor(0.0, device=DEVICE)
                    loss_boundary = torch.tensor(0.0, device=DEVICE)
                else:
                    loss_ce = criterion(outputs, yb)
                    loss_conf = confusion_penalty(outputs, yb)
                    loss_n2 = n2_focus_loss(outputs, yb)
                    loss_n3 = n3_focus_loss(outputs, yb)
                    loss_prec = precision_loss(outputs, yb)
                    loss_n3_prec = n3_precision_loss(outputs, yb)
                    loss_rem_prec = rem_precision_loss(outputs, yb)
                    loss_boundary = boundary_loss(outputs, yb)

                ls = spatial_graph_loss_seq(SG_seq, Xn_seq, lambda_frob=1e-3, lambda_smooth=1e-3)
                lt = temporal_graph_loss(TG, epoch_seq, lambda_frob=1e-3, lambda_smooth=1e-3)

                loss = (loss_ce + loss_conf + loss_n2 + loss_n3 + loss_prec + 
                       loss_n3_prec + loss_rem_prec + loss_boundary +
                       LAMBDA_S * ls + LAMBDA_T * lt)
                loss = loss / GRAD_ACCUM_STEPS

            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                gp = sum((p ** 2).sum() for p in list(model.parameters()) + list(raw_projector.parameters()) if p.requires_grad)
                total_loss = loss + LAMBDA_GP * gp / GRAD_ACCUM_STEPS
            else:
                total_loss = loss

            scaler.scale(total_loss).backward()

            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(raw_projector.parameters()), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * GRAD_ACCUM_STEPS
            if not use_mixup:
                confusion_loss_accum += loss_conf.item() if torch.is_tensor(loss_conf) else float(loss_conf)
                n3_prec_loss_accum += loss_n3_prec.item() if torch.is_tensor(loss_n3_prec) else float(loss_n3_prec)

            if not use_mixup:
                preds = outputs.argmax(dim=1)
                train_preds.extend(preds.cpu().numpy())
                train_targets.extend(yb.cpu().numpy())

            if torch.cuda.is_available() and batch_idx % 50 == 0:
                torch.cuda.empty_cache()

        train_loss /= max(1, len(train_loader))
        confusion_loss_avg = confusion_loss_accum / max(1, len(train_loader))
        n3_prec_loss_avg = n3_prec_loss_accum / max(1, len(train_loader))

        train_f1 = f1_score(train_targets, train_preds, average='macro', zero_division=0) if len(train_preds) > 0 else 0.0

        if epoch <= WARMUP_EPOCHS:
            warmup_scheduler.step()
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        # Validation
        model.eval()
        raw_projector.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        class_correct = defaultdict(int)
        class_total = defaultdict(int)

        with torch.no_grad():
            for Xb, yb, sids in val_loader:
                rawb = raw_projector(Xb)
                outputs, SG_seq, TG, Xn_seq, g_out_seq, epoch_seq = model(Xb, rawb)
                loss_ce = criterion(outputs, yb)
                ls = spatial_graph_loss_seq(SG_seq, Xn_seq, lambda_frob=1e-3, lambda_smooth=1e-3)
                lt = temporal_graph_loss(TG, epoch_seq, lambda_frob=1e-3, lambda_smooth=1e-3)
                loss = loss_ce + LAMBDA_S * ls + LAMBDA_T * lt

                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(yb.cpu().numpy())

                for pred, target in zip(preds.cpu().numpy(), yb.cpu().numpy()):
                    class_total[target] += 1
                    if pred == target:
                        class_correct[target] += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        val_loss /= max(1, len(val_loader))
        val_f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0) if len(val_preds) > 0 else 0.0

        class_f1_scores = f1_score(val_targets, val_preds, average=None, labels=range(NUM_CLASSES), zero_division=0)
        criterion.update_gammas(class_f1_scores)

        improved, stop_flag = monitor.update(train_loss, val_loss, train_f1, val_f1, epoch, current_lr, confusion_loss_avg, n3_prec_loss_avg)

        if val_f1 > (best_meta.get('best_val_f1', -1e9)):
            best_meta['best_val_f1'] = val_f1
            best_meta['epoch'] = epoch
            best_state = {
                'model_state': {k: v.cpu().clone() for k, v in model.state_dict().items()},
                'projector_state': {k: v.cpu().clone() for k, v in raw_projector.state_dict().items()},
                'optimizer_state': optimizer.state_dict(),
                'epoch': epoch,
                'monitor': monitor.history
            }

        elapsed = time.time() - epoch_start

        overfit_gap = float(train_f1 - val_f1)
        overfit_window.append(overfit_gap > OVERFIT_THRESHOLD)
        if sum(overfit_window) == OVERFIT_PATIENCE:
            print(f"*** Overfitting detected. Reducing LR.")
            reduce_on_plateau.step(val_loss)
            def increase_dropout(module, inc=0.05, maxv=0.55):
                for child in module.modules():
                    if isinstance(child, nn.Dropout):
                        child.p = min(maxv, child.p + inc)
            increase_dropout(model.spatial_module, inc=0.03)
            increase_dropout(model.fusion, inc=0.03)
            overfit_window.clear()

        recalls_str = " ".join([f"R{i}:{(class_correct[i]/max(1,class_total[i])):.2f}" for i in range(NUM_CLASSES)])
        f1_str = " ".join([f"F{i}:{class_f1_scores[i]:.2f}" for i in range(NUM_CLASSES)])

        if torch.cuda.is_available():
            gpu_mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"Ep {epoch:03d} | TrLs {train_loss:.4f} VaLs {val_loss:.4f} | Conf {confusion_loss_avg:.3f} N3Pr {n3_prec_loss_avg:.3f} | TrF1 {train_f1:.4f} VaF1 {val_f1:.4f} | LR {current_lr:.2e} | {recalls_str} | {f1_str} | GPU {gpu_mem:.2f}GB | {elapsed:.0f}s")
            torch.cuda.reset_peak_memory_stats()
        else:
            print(f"Ep {epoch:03d} | TrLs {train_loss:.4f} VaLs {val_loss:.4f} | Conf {confusion_loss_avg:.3f} N3Pr {n3_prec_loss_avg:.3f} | TrF1 {train_f1:.4f} VaF1 {val_f1:.4f} | LR {current_lr:.2e} | {recalls_str} | {f1_str} | {elapsed:.0f}s")

        if stop_flag:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state['model_state'])
        raw_projector.load_state_dict(best_state['projector_state'])

    checkpoint = {
        'model_state': {k: v.cpu() for k, v in model.state_dict().items()},
        'projector_state': {k: v.cpu() for k, v in raw_projector.state_dict().items()},
        'monitor': monitor.history,
        'best_meta': best_meta
    }
    torch.save(checkpoint, MODEL_OUT)
    print(f"\nSaved best model to {MODEL_OUT} (best epoch {best_meta.get('epoch','N/A')}, best val F1 {best_meta.get('best_val_f1','N/A'):.4f})")

    plot_training_curves(monitor)
    print(f"Saved training curves to {TRAIN_CURVES}")

    # Test evaluation with calibrated thresholds
    y_pred = []
    y_pred_calibrated = []
    y_true = []
    SG_collect = []
    TG_collect = []
    labels_collect = []

    model.eval()
    raw_projector.eval()
    with torch.no_grad():
        for Xb, yb, sids in test_loader:
            rawb = raw_projector(Xb)
            outputs, SG_seq, TG, Xn_seq, g_out_seq, epoch_seq = model(Xb, rawb)
            
            # Standard predictions
            preds = outputs.argmax(dim=1)
            y_pred.extend(preds.cpu().numpy())
            
            # Calibrated predictions
            preds_calibrated = apply_calibrated_thresholds(outputs, CLASS_THRESHOLDS)
            y_pred_calibrated.extend(preds_calibrated.cpu().numpy())
            
            y_true.extend(yb.cpu().numpy())

            SG_avg = SG_seq.mean(dim=1).cpu().numpy()
            TG_np = TG.cpu().numpy()
            SG_collect.append(SG_avg)
            TG_collect.append(TG_np)
            labels_collect.append(yb.cpu().numpy())

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    y_pred = np.array(y_pred)
    y_pred_calibrated = np.array(y_pred_calibrated)
    y_true = np.array(y_true)
    SG_collect = np.concatenate(SG_collect, axis=0) if len(SG_collect) > 0 else np.zeros((0, 6, 6))
    TG_collect = np.concatenate(TG_collect, axis=0) if len(TG_collect) > 0 else np.zeros((0, T, T))
    labels_collect = np.concatenate(labels_collect, axis=0) if len(labels_collect) > 0 else np.zeros((0,))

    print("\n" + "="*60)
    print("FINAL TEST METRICS (Standard)")
    print("="*60)
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro-F1: {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Weighted-F1: {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"Cohen's Kappa: {cohen_kappa_score(y_true, y_pred):.4f}")
    print("\n" + classification_report(y_true, y_pred, target_names=['Wake', 'N1', 'N2', 'N3', 'REM'], zero_division=0))

    print("\n" + "="*60)
    print("FINAL TEST METRICS (Calibrated Thresholds)")
    print("="*60)
    print(f"Accuracy: {accuracy_score(y_true, y_pred_calibrated):.4f}")
    print(f"Macro-F1: {f1_score(y_true, y_pred_calibrated, average='macro', zero_division=0):.4f}")
    print(f"Weighted-F1: {f1_score(y_true, y_pred_calibrated, average='weighted', zero_division=0):.4f}")
    print(f"Cohen's Kappa: {cohen_kappa_score(y_true, y_pred_calibrated):.4f}")
    print("\n" + classification_report(y_true, y_pred_calibrated, target_names=['Wake', 'N1', 'N2', 'N3', 'REM'], zero_division=0))

    # Use calibrated predictions for confusion matrix
    cm = confusion_matrix(y_true, y_pred_calibrated)
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    labels = ['Wake', 'N1', 'N2', 'N3', 'REM']
    sns.heatmap(cm, annot=True, fmt='d', ax=ax1, xticklabels=labels, yticklabels=labels, cmap='Blues')
    ax1.set_title('Confusion Matrix (Counts) - Calibrated')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    sns.heatmap(cm_norm, annot=True, fmt='.3f', ax=ax2, xticklabels=labels, yticklabels=labels, vmin=0, vmax=1, cmap='RdYlGn')
    ax2.set_title('Confusion Matrix (Recall Normalized) - Calibrated')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(CONF_MAT, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {CONF_MAT}")

    # Edge visualizations
    os.makedirs(EDGE_VIZ_DIR, exist_ok=True)
    class_sums = defaultdict(lambda: np.zeros((6, 6), dtype=float))
    class_counts = defaultdict(int)
    overall_SG_mean = SG_collect.mean(axis=0) if SG_collect.shape[0] > 0 else np.zeros((6, 6))
    overall_TG_mean = TG_collect.mean(axis=0) if TG_collect.shape[0] > 0 else np.zeros((T, T))

    if len(SG_collect) > 0:
        for batch_idx in range(len(SG_collect)):
            sg_batch = SG_collect[batch_idx]
            lbl_batch = labels_collect[batch_idx] if batch_idx < len(labels_collect) else None
            if lbl_batch is not None:
                lbl = int(lbl_batch) if np.isscalar(lbl_batch) else int(lbl_batch[0])
                class_sums[lbl] += sg_batch
                class_counts[lbl] += 1

    for cls in range(NUM_CLASSES):
        if class_counts[cls] > 0:
            mat = class_sums[cls] / float(class_counts[cls])
        else:
            mat = np.zeros((6, 6))
        title = f"Avg Spatial Adjacency - cls {cls} (n={class_counts[cls]})"
        fname = os.path.join(EDGE_VIZ_DIR, f"avg_SG_class_{cls}.png")
        ch_labels = channel_names if channel_names is not None else [f"Ch{i}" for i in range(6)]
        save_adj_heatmap(mat, title, fname, xticklabels=ch_labels, yticklabels=ch_labels, vmin=0.0, vmax=1.0)

    fname = os.path.join(EDGE_VIZ_DIR, "avg_SG_overall.png")
    ch_labels = channel_names if channel_names is not None else [f"Ch{i}" for i in range(6)]
    save_adj_heatmap(overall_SG_mean, "Average Spatial Adjacency - All Classes", fname, xticklabels=ch_labels, yticklabels=ch_labels, vmin=0.0, vmax=1.0)

    fname = os.path.join(EDGE_VIZ_DIR, "avg_TG_overall.png")
    t_labels = [f"t-{T//2-i}" if i < T//2 else f"t+{i-T//2}" for i in range(T)]
    save_adj_heatmap(overall_TG_mean, "Average Temporal Adjacency - All Classes", fname, xticklabels=t_labels, yticklabels=t_labels, vmin=0.0, vmax=1.0)

    print(f"Saved edge visualizations to {EDGE_VIZ_DIR}/")
    return model, y_pred_calibrated, y_true, monitor

# ------------------------------ Main ------------------------------
if __name__ == "__main__":
    if not Path(DATA_NPZ).exists():
        raise FileNotFoundError(f"Preprocessed NPZ not found: {DATA_NPZ}")
    print("="*60)
    print("IMPROVED STGNN SLEEP STAGING TRAINING V2 - RESEARCH GRADE")
    print("REQUIREMENTS:")
    print("- 6 channels (EEG Fpz-Cz, EEG Pz-Oz, EOG, EMG, Resp, Temp)")
    print("- 5 canonical frequency bands (Delta, Theta, Alpha, Beta, Gamma)")
    print("- Temporal receptive field: 9 epochs")
    print("\nKey Improvements:")
    print("- Multi-modal physiological signals (brain, eyes, muscles, breathing, temperature)")
    print("- Enhanced N3/REM precision losses")
    print("- N1/N2 boundary discrimination")
    print("- Calibrated thresholds for minority classes")
    print("- Increased class weights for N1/N3")
    print("="*60)
    model, y_pred, y_true, monitor = train_pipeline(DATA_NPZ)
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best epoch: {monitor.best_epoch}")
    print(f"Best val F1: {monitor.best_val_f1:.4f}")
    print(f"Best val loss: {monitor.best_val_loss:.4f}")
    print("="*60)