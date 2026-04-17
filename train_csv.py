"""
DPINet adapted for electrode calendering CSV data.

Calendering physics:
- Compression axis (COMPRESS_AXIS): the plate presses down along this axis
- Bottom plate: fixed — particles near the bottom should barely move
- Top plate: moving — particles near the top are pressed most

Physics are encoded via:
  1. Extended attr = [norm_radius, height_norm, near_bottom, near_top]
     so the model sees each particle's role in the stack.
  2. A boundary loss that penalises bottom-particle velocity along the
     compression axis during training.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from types import SimpleNamespace

import scipy.spatial as spatial
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from models import DPINet

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
def _select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE  = _select_device()
USE_GPU = DEVICE.type != "cpu"   # passed to DPINet to trigger device-aware init

DATA_DIR = Path("Data")
OUTPUT_DIR = Path("Results")
MODEL_DIR = Path("Models")
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z',
            'ang_vel_x', 'ang_vel_y', 'ang_vel_z', 'radii']

ALL_FILES = [
    "particle_dynamics_full_data_speed_1e-04ms.csv",
    "particle_dynamics_full_data_speed_0.001ms.csv",
    "particle_dynamics_full_data_speed_0.01ms.csv",
    "particle_dynamics_full_data_speed_0.1ms.csv",
    "particle_dynamics_full_data_speed_1ms.csv",
    "particle_dynamics_full_data_speed_5ms.csv",
]

TEST_INDEX = 3
TRAIN_FILES = [f for i, f in enumerate(ALL_FILES) if i != TEST_INDEX]
TEST_FILE = ALL_FILES[TEST_INDEX]

KNN_K = 16
EPOCHS = 50
LR = 1e-4

# ── Calendering physics ────────────────────────────────────────────────────────
# Which position axis is the compression direction: 0=x, 1=y, 2=z
COMPRESS_AXIS = 2

# Fraction of total stack height considered "near bottom" / "near top"
BOTTOM_THRESHOLD = 0.15
TOP_THRESHOLD    = 0.15

# Weight applied to the bottom-particle boundary loss
LAMBDA_BOUNDARY = 5.0
# ──────────────────────────────────────────────────────────────────────────────

# DPINet model dimensions — must stay consistent with models.py formulas:
#   ParticleEncoder  input = attr_dim + state_dim * 2
#   RelationEncoder  input = 2*attr_dim + 4*state_dim + relation_dim
#
# DPINet.forward() appends a state_dim-sized offset vector to attr before
# calling the encoders, so the effective attr size is (attr_dim + state_dim).
# The formulas above already account for this.
ARGS = SimpleNamespace(
    state_dim=6,        # [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z]
    attr_dim=4,         # [norm_radius, height_norm, near_bottom, near_top]
    relation_dim=1,     # dummy edge feature
    position_dim=3,     # output: predicted next velocity (x, y, z)
    nf_particle=200,
    nf_relation=300,
    nf_effect=200,
    n_stages=1,
    pstep=2,
    dt=1.0,             # only used for rigid bodies — ignored here
    n_his=0,
)

PHASES_DICT = {
    'material': ['fluid'],
    'instance': ['fluid'],
    'root_num': [[]],
}

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def get_simulation_matrix(filename):
    df = pd.read_csv(DATA_DIR / filename)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
    df = df.drop_duplicates(subset=['time', 'particleID'], keep='first')
    df = df.sort_values(['time', 'particleID'])

    pids  = sorted(df['particleID'].unique())
    times = sorted(df['time'].unique())

    matrix = np.zeros((10, len(times), len(pids)), dtype=np.float32)
    for i, feat in enumerate(FEATURES):
        pivot = df.pivot(index='time', columns='particleID', values=feat)
        pivot = pivot.interpolate(method='linear', axis=0).ffill().bfill()
        matrix[i] = pivot.values

    return matrix, pids, times

# ─────────────────────────────────────────────
# STATISTICS
# ─────────────────────────────────────────────
def compute_statistics(matrices):
    """
    Compute per-dimension mean/std for positions, velocities, and radii.
    Returns:
        stat       : list of two [3, 3] arrays [[mean, std, count]] for pos and vel
        radii_stat : [mean, std] scalar pair
    """
    pos_acc, vel_acc, rad_acc = [], [], []
    for mat in matrices:
        pos_acc.append(mat[:3].reshape(3, -1).T)
        vel_acc.append(mat[3:6].reshape(3, -1).T)
        rad_acc.append(mat[9].reshape(-1))

    all_pos = np.concatenate(pos_acc, axis=0)
    all_vel = np.concatenate(vel_acc, axis=0)
    all_rad = np.concatenate(rad_acc, axis=0)

    def _stat3(arr):
        mean = arr.mean(0)
        std  = arr.std(0);  std[std == 0] = 1.0
        n    = np.full(3, len(arr), dtype=np.float32)
        return np.stack([mean, std, n], axis=1)   # [3, 3]

    stat = [_stat3(all_pos), _stat3(all_vel)]

    rad_mean = float(all_rad.mean())
    rad_std  = float(all_rad.std());  rad_std = rad_std if rad_std > 0 else 1.0
    radii_stat = np.array([rad_mean, rad_std], dtype=np.float32)

    return stat, radii_stat

# ─────────────────────────────────────────────
# GRAPH CONSTRUCTION
# ─────────────────────────────────────────────
def build_relations(pos, k):
    """
    KNN graph in DPINet's sparse-matrix format.
    Returns (Rr, Rs, Ra, node_r_idx, node_s_idx) or None if no edges.
    """
    N    = pos.shape[0]
    tree = spatial.cKDTree(pos)
    _, nbrs = tree.query(pos, k=min(k + 1, N))

    pairs = [(i, int(j)) for i in range(N) for j in nbrs[i] if j != i]
    if not pairs:
        return None

    rels   = np.array(pairs, dtype=np.int64)
    n_rels = len(rels)

    # Dense matrices: sparse ops are not supported on MPS and are optional on CPU.
    # DPINet's .t().mm() calls work identically with dense tensors.
    Rr = torch.zeros(N, n_rels)
    Rs = torch.zeros(N, n_rels)
    Rr[rels[:, 0], np.arange(n_rels)] = 1.0
    Rs[rels[:, 1], np.arange(n_rels)] = 1.0
    Ra = torch.zeros(n_rels, ARGS.relation_dim)

    return Rr, Rs, Ra, np.arange(N), np.arange(N)

# ─────────────────────────────────────────────
# CALENDERING GEOMETRY HELPERS
# ─────────────────────────────────────────────
def calendering_attr(pos, radii, radii_stat):
    """
    Build the 4-column attr array for one timestep.

    Columns: [norm_radius, height_norm, near_bottom, near_top]
      height_norm  : 0 = bottom plate, 1 = top plate
      near_bottom  : 1 if particle is in the bottom BOTTOM_THRESHOLD fraction
      near_top     : 1 if particle is in the top    TOP_THRESHOLD fraction
    """
    z      = pos[:, COMPRESS_AXIS]           # compression-axis coordinate
    z_min, z_max = z.min(), z.max()
    z_range = max(float(z_max - z_min), 1e-8)

    height_norm  = (z - z_min) / z_range    # [N]  in [0, 1]
    near_bottom  = (height_norm < BOTTOM_THRESHOLD).astype(np.float32)
    near_top     = (height_norm > (1.0 - TOP_THRESHOLD)).astype(np.float32)
    norm_radii   = (radii - radii_stat[0]) / radii_stat[1]

    return np.stack([norm_radii, height_norm.astype(np.float32),
                     near_bottom, near_top], axis=1)   # [N, 4]

# ─────────────────────────────────────────────
# PER-STEP PREPARATION
# ─────────────────────────────────────────────
def prepare_step(mat_t, mat_t1, stat, radii_stat):
    """
    Build tensors for one timestep transition.

    Returns: attr_t, state_t, rels, N, label_t, bottom_mask
      attr_t      : [N, 4]  calendering geometry attributes
      state_t     : [N, 6]  normalised [pos, vel]
      label_t     : [N, 3]  normalised next velocity
      bottom_mask : [N]     bool, True for particles near the fixed bottom plate
    """
    pos_mean, pos_std = stat[0][:, 0], stat[0][:, 1]
    vel_mean, vel_std = stat[1][:, 0], stat[1][:, 1]

    pos      = mat_t[:3].T     # [N, 3]
    vel      = mat_t[3:6].T    # [N, 3]
    radii    = mat_t[9]        # [N]
    next_vel = mat_t1[3:6].T   # [N, 3]

    norm_pos      = (pos      - pos_mean) / pos_std
    norm_vel      = (vel      - vel_mean) / vel_std
    norm_next_vel = (next_vel - vel_mean) / vel_std

    state = np.concatenate([norm_pos, norm_vel], axis=1).astype(np.float32)
    attr  = calendering_attr(pos, radii, radii_stat)
    label = norm_next_vel.astype(np.float32)

    bottom_mask = attr[:, 2].astype(bool)   # near_bottom column

    rels = build_relations(pos, KNN_K)

    return (
        torch.FloatTensor(attr),
        torch.FloatTensor(state),
        rels,
        pos.shape[0],
        torch.FloatTensor(label),
        torch.BoolTensor(bottom_mask),
    )

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
def _to_device(tensors):
    return [t.to(DEVICE) for t in tensors]

def train(model, matrices, stat, radii_stat):
    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=3, verbose=True)
    criterion = nn.MSELoss()
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        count = 0

        for mat in matrices:
            _, T, _ = mat.shape

            for t in range(T - 1):
                attr_t, state_t, rels, N, label_t, bottom_mask = prepare_step(
                    mat[:, t, :], mat[:, t + 1, :], stat, radii_stat
                )
                if rels is None:
                    continue

                Rr, Rs, Ra, node_r_idx, node_s_idx = rels
                attr_t, state_t, Ra, label_t, Rr, Rs = _to_device(
                    [attr_t, state_t, Ra, label_t, Rr, Rs]
                )

                optimizer.zero_grad()
                pred = model(
                    attr_t, state_t,
                    [Rr], [Rs], [Ra],
                    N,
                    [node_r_idx], [node_s_idx],
                    [ARGS.pstep],
                    [0, N], PHASES_DICT,
                )

                loss = criterion(pred, label_t)

                # Physics constraint: bottom particles should not move along
                # the compression axis (fixed plate boundary condition).
                if bottom_mask.any():
                    bottom_idx = bottom_mask.nonzero(as_tuple=False).squeeze(1)
                    if USE_GPU:
                        bottom_idx = bottom_idx.to(DEVICE)
                    bottom_compress_vel = pred[bottom_idx, COMPRESS_AXIS]
                    loss = loss + LAMBDA_BOUNDARY * bottom_compress_vel.pow(2).mean()

                if torch.isnan(loss):
                    print(f"  NaN at t={t}, skipping")
                    continue

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                count += 1

        avg_loss = total_loss / max(count, 1)
        print(f"Epoch {epoch + 1}/{EPOCHS}  avg_loss={avg_loss:.6f}")
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_DIR / "dpi_best.pth")

        torch.save(model.state_dict(), MODEL_DIR / f"dpi_epoch_{epoch + 1}.pth")

# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
def inference(model, mat, times, stat, radii_stat):
    model.eval()
    pos_mean, pos_std = stat[0][:, 0], stat[0][:, 1]
    vel_mean, vel_std = stat[1][:, 0], stat[1][:, 1]

    curr_pos   = mat[:3,  0, :].T.copy()   # [N, 3]
    curr_vel   = mat[3:6, 0, :].T.copy()   # [N, 3]
    curr_radii = mat[9,   0, :].copy()      # [N]
    results = []

    for t in tqdm(range(len(times) - 1)):
        norm_pos = (curr_pos - pos_mean) / pos_std
        norm_vel = (curr_vel - vel_mean) / vel_std
        attr_np  = calendering_attr(curr_pos, curr_radii, radii_stat)

        state_t = torch.FloatTensor(np.concatenate([norm_pos, norm_vel], axis=1))
        attr_t  = torch.FloatTensor(attr_np)

        rels = build_relations(curr_pos, KNN_K)
        if rels is None:
            continue
        Rr, Rs, Ra, node_r_idx, node_s_idx = rels
        N = curr_pos.shape[0]

        state_t, attr_t, Ra, Rr, Rs = _to_device([state_t, attr_t, Ra, Rr, Rs])

        with torch.no_grad():
            pred_norm = model(
                attr_t, state_t,
                [Rr], [Rs], [Ra],
                N,
                [node_r_idx], [node_s_idx],
                [ARGS.pstep],
                [0, N], PHASES_DICT,
            )

        pred_vel = pred_norm.cpu().numpy() * vel_std + vel_mean   # [N, 3]
        dt = times[t + 1] - times[t]
        next_pos = curr_pos + pred_vel * dt

        near_bottom = attr_np[:, 2].astype(bool)
        for i, p in enumerate(next_pos):
            results.append({
                'time':        times[t + 1],
                'x':           p[0],
                'y':           p[1],
                'z':           p[2],
                'near_bottom': bool(near_bottom[i]),
                'near_top':    bool(attr_np[i, 3]),
            })

        curr_pos = next_pos
        curr_vel = pred_vel

    return pd.DataFrame(results)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    print(f"Compress axis: {'xyz'[COMPRESS_AXIS]}  "
          f"bottom_threshold={BOTTOM_THRESHOLD}  top_threshold={TOP_THRESHOLD}  "
          f"lambda_boundary={LAMBDA_BOUNDARY}")

    print("\nLoading training data...")
    matrices = []
    for f in TRAIN_FILES:
        mat, _, _ = get_simulation_matrix(f)
        matrices.append(mat)
        print(f"  {f}  shape={mat.shape}")

    print("\nComputing statistics...")
    stat, radii_stat = compute_statistics(matrices)
    np.save(MODEL_DIR / "stat.npy", np.array(stat, dtype=object), allow_pickle=True)
    np.save(MODEL_DIR / "radii_stat.npy", radii_stat)

    model = DPINet(ARGS, stat, PHASES_DICT, residual=True, use_gpu=USE_GPU)
    model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"DPINet parameters: {n_params:,}")

    train(model, matrices, stat, radii_stat)

    print("\nInference on test file...")
    test_mat, _, times = get_simulation_matrix(TEST_FILE)

    model.load_state_dict(torch.load(MODEL_DIR / "dpi_best.pth", map_location=DEVICE))
    df = inference(model, test_mat, list(times), stat, radii_stat)

    out_path = OUTPUT_DIR / "dpi_results.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print(f"  Bottom particles: {df['near_bottom'].sum()}")
    print(f"  Top particles:    {df['near_top'].sum()}")


if __name__ == "__main__":
    main()
