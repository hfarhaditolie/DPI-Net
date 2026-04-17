import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from torch_geometric.nn import MessagePassing
from torch_cluster import knn_graph

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = Path("Data")
OUTPUT_DIR = Path("Results")
MODEL_DIR = Path("Models")

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 2048
LR = 1e-3
EPOCHS = 50
NUM_SAMPLES = 1024
KNN_K = 16

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

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def get_simulation_matrix(filename):
    df = pd.read_csv(DATA_DIR / filename)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
    df = df.drop_duplicates(subset=['time', 'particleID'], keep='first')
    df = df.sort_values(['time', 'particleID'])

    pids = sorted(df['particleID'].unique())
    times = sorted(df['time'].unique())

    matrix = np.zeros((10, len(times), len(pids)), dtype=np.float32)

    for i, feat in enumerate(FEATURES):
        pivot_df = df.pivot(index='time', columns='particleID', values=feat)
        pivot_df = pivot_df.interpolate(method='linear', axis=0).ffill().bfill()
        matrix[i, :, :] = pivot_df.values

    return torch.from_numpy(matrix), pids, times

# ─────────────────────────────────────────────
# SAMPLING + TARGETS
# ─────────────────────────────────────────────
def sample_query_points(pos, num_samples):
    min_p = pos.min(0)[0]
    max_p = pos.max(0)[0]
    return torch.rand(num_samples, 3, device=pos.device) * (max_p - min_p) + min_p

def compute_targets(query_pts, current_pos, next_pos, radius=0.01):
    dist = torch.cdist(query_pts, current_pos)
    nn_idx = dist.argmin(dim=1)
    nn_dist = dist.min(dim=1)[0]

    occ = (nn_dist < radius).float().unsqueeze(1)
    disp = next_pos[nn_idx] - current_pos[nn_idx]

    return disp, occ

# ─────────────────────────────────────────────
# GNN MODEL
# ─────────────────────────────────────────────
class GNNLayer(MessagePassing):
    def __init__(self, in_dim, out_dim):
        super().__init__(aggr='add')
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        return self.mlp(torch.cat([x_i, x_j], dim=-1))

class FieldGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = GNNLayer(10, 64)
        self.layer2 = GNNLayer(64, 64)

        self.head_disp = nn.Linear(64, 3)
        self.head_occ  = nn.Linear(64, 1)

    def forward(self, x, edge_index):
        x = F.silu(self.layer1(x, edge_index))
        x = F.silu(self.layer2(x, edge_index))

        disp = self.head_disp(x)
        occ  = torch.sigmoid(self.head_occ(x))

        return disp, occ

# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────
def train(model, raw_mats):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        print(f"\nEpoch {epoch+1}")

        for mat in raw_mats:
            _, T, N = mat.shape

            for t in range(T - 1):
                current = mat[:, t, :].T.to(DEVICE)
                next_f  = mat[:, t+1, :].T.to(DEVICE)

                pos = current[:, :3]
                next_pos = next_f[:, :3]

                query = sample_query_points(pos, NUM_SAMPLES)

                disp_gt, occ_gt = compute_targets(query, pos, next_pos)

                # Build graph (particles + queries)
                all_pos = torch.cat([pos, query], dim=0)
                edge_index = knn_graph(all_pos, k=KNN_K)

                # Features: particles = real features, queries = zeros
                particle_feat = current
                query_feat = torch.zeros(NUM_SAMPLES, 10, device=DEVICE)

                x = torch.cat([particle_feat, query_feat], dim=0)

                optimizer.zero_grad()

                pred_disp, pred_occ = model(x, edge_index)

                # only query nodes
                pred_disp_q = pred_disp[N:]
                pred_occ_q  = pred_occ[N:]

                loss_disp = F.huber_loss(pred_disp_q, disp_gt)
                loss_occ  = F.binary_cross_entropy(pred_occ_q, occ_gt)

                loss = loss_disp + 0.5 * loss_occ

                if torch.isnan(loss):
                    print("NaN loss, skipping")
                    continue

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                print(f"  t={t} loss: {loss.item():.6f}")

        print(f"Epoch {epoch+1} total loss: {total_loss:.6f}")

        torch.save(model.state_dict(), MODEL_DIR / f"gnn_epoch_{epoch+1}.pth")

# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
def inference(model, mat, times):
    model.eval()

    current = mat[:, 0, :].T.to(DEVICE)
    pos = current[:, :3]

    results = []

    for t in tqdm(range(len(times)-1)):
        query = sample_query_points(pos, 5000)

        all_pos = torch.cat([pos, query], dim=0)
        edge_index = knn_graph(all_pos, k=KNN_K)

        particle_feat = current
        query_feat = torch.zeros(5000, 10, device=DEVICE)

        x = torch.cat([particle_feat, query_feat], dim=0)

        with torch.no_grad():
            disp, occ = model(x, edge_index)

        disp_q = disp[pos.shape[0]:]
        occ_q  = occ[pos.shape[0]:]

        mask = occ_q.squeeze() > 0.5
        pred_pos = query[mask] + disp_q[mask]

        for p in pred_pos.cpu().numpy():
            results.append({
                "time": times[t+1],
                "x": p[0],
                "y": p[1],
                "z": p[2]
            })

    return pd.DataFrame(results)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print(f"Using device: {DEVICE}")

    print("Loading training data...")
    raw_mats = [get_simulation_matrix(f)[0] for f in TRAIN_FILES]

    model = FieldGNN().to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    train(model, raw_mats)

    print("\nTesting...")
    test_mat, _, times = get_simulation_matrix(TEST_FILE)

    df = inference(model, test_mat, times)

    out_path = OUTPUT_DIR / "gnn_field_results.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
