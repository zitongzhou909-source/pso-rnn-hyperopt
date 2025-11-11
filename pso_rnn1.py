# ============================================
# Suitable for: M4 Monthly Macro single series; Direct 18-step forecast
# ============================================

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

# -----------------------------
# Original time series (M4 Macro monthly series)
# -----------------------------
RAW_SERIES = [
    #This is the original time series used for forecasting. Some data are omitted here for brevity.
    8000,8350,8570,7700,7080,6520,6070,6650,6830,5710,5260,5470,7870,7360,8470,7880,6750,6860,6220,6650,...
    #  

# -----------------------------
# Utility functions
# -----------------------------
def series_train_test_split(y, horizon=18):
    """Split series into training and test parts."""
    y = np.asarray(y, dtype=float)
    assert len(y) > horizon, "Series too short to split."
    return y[:-horizon], y[-horizon:]

def zscore_fit_transform(train, *others):
    """Standardize the series using training mean and std."""
    mu, sigma = float(np.mean(train)), float(np.std(train))
    if sigma < 1e-12: sigma = 1.0
    def tf(x): return (np.asarray(x, dtype=float) - mu) / sigma
    outs = [tf(train)] + [tf(o) for o in others]
    return (mu, sigma), outs

def zscore_inverse(x, mu, sigma):
    """Reverse z-score normalization."""
    return np.asarray(x)*sigma + mu

# -----------------------------
# Dataset and Model
# -----------------------------
class WindowDataset(Dataset):
    """Dataset that creates sliding windows of input/output pairs."""
    def __init__(self, y, window, horizon=18):
        self.y = np.asarray(y, dtype=float)
        self.window = int(window)
        self.horizon = int(horizon)
        self.X, self.Y = [], []
        N = len(self.y) - self.window - self.horizon + 1
        if N <= 0:
            print(f"[Warning] Cannot create samples (window={window}, horizon={horizon}, len(y)={len(self.y)})")
        else:
            for t in range(N):
                self.X.append(self.y[t:t+self.window])
                self.Y.append(self.y[t+self.window:t+self.window+self.horizon])
        self.X = torch.tensor(self.X, dtype=torch.float32).unsqueeze(-1)  # [N, W, 1]
        self.Y = torch.tensor(self.Y, dtype=torch.float32)               # [N, 18]
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.Y[i]

class RNNDirect18(nn.Module):
    """Simple RNN predicting next 18 steps."""
    def __init__(self, hidden_size, num_layers, dropout):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity='tanh',
            dropout=(dropout if num_layers > 1 else 0.0)
        )
        self.head = nn.Linear(hidden_size, 18)
    def forward(self, x):
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.head(last)

# -----------------------------
# Validation split (last 20% of samples)
# -----------------------------
def split_windows_for_val(ds, val_ratio=0.2):
    """Split dataset into training and validation subsets."""
    n = len(ds)
    if n == 0:
        return Subset(ds, []), Subset(ds, [])
    k = max(int(round(n*(1-val_ratio))), 1)
    idx = list(range(n))
    return Subset(ds, idx[:k]), Subset(ds, idx[k:])

# -----------------------------
# Train one model and return validation MAE and best state
# -----------------------------
def train_one_model(train_y, window, lr, hidden, layers, dropout, epochs=60, batch_size=64, device=None):
    """Train one RNN model and return its validation MAE."""
    H = 18
    L = len(train_y)
    max_valid_window = L - H
    if window > max_valid_window:
        print(f"[Invalid particle] window={window} > max_valid_window={max_valid_window} (len(train)={L}, horizon=18)")
        return 1e12, {"reason": "window_too_large"}

    (mu, sigma), [train_z] = zscore_fit_transform(train_y)
    ds = WindowDataset(train_z, window, H)
    if len(ds) == 0:
        return 1e12, {"reason": "no_samples"}

    tr_sub, val_sub = split_windows_for_val(ds, 0.2)
    if len(val_sub) == 0:
        print("[Warning] No validation samples. Returning large penalty.")
        return 1e12, {"reason": "no_val"}

    tr_loader = DataLoader(tr_sub, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = RNNDirect18(hidden, layers, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()

    best_val = float("inf")
    best_state = None

    for ep in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        # Validation
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb.to(device)).cpu().numpy()
                val_preds.append(pred)
                val_trues.append(yb.numpy())

        if len(val_preds) == 0:
            print("[Validation] No batches collected. Returning large penalty.")
            return 1e12, {"reason": "no_val_batches"}

        val_preds = np.concatenate(val_preds, axis=0)
        val_trues = np.concatenate(val_trues, axis=0)
        val_preds_inv = zscore_inverse(val_preds, mu, sigma)
        val_trues_inv = zscore_inverse(val_trues, mu, sigma)
        mae = float(np.mean(np.abs(val_preds_inv - val_trues_inv)))

        if mae < best_val:
            best_val = mae
            best_state = {
                "window": int(window),
                "hidden": int(hidden),
                "layers": int(layers),
                "dropout": float(dropout),
                "lr": float(lr),
                "mu": float(mu),
                "sigma": float(sigma),
                "model_state": model.state_dict()
            }

    if best_state is None:
        return 1e12, {"reason": "no_improvement"}
    return best_val, best_state

# -----------------------------
# Particle Swarm Optimization (PSO)
# -----------------------------
class PSOConfig:
    """PSO configuration parameters."""
    particles = 8
    iters = 10
    w = 0.7
    c1 = 1.4
    c2 = 1.4
    seed = 42

def pso_optimize(train_y, device=None):
    """Perform PSO-based hyperparameter search."""
    rng = np.random.default_rng(PSOConfig.seed)

    def init_particle():
        # [lr_log10, hidden, window, dropout, layers]
        return np.array([
            rng.uniform(-4, -2),
            rng.integers(32, 257),
            rng.integers(6, 37),
            rng.uniform(0.0, 0.5),
            rng.integers(1, 3)
        ], dtype=float)

    def decode(pos):
        lr = 10**pos[0]
        hidden = int(np.clip(round(pos[1]), 32, 256))
        window = int(np.clip(round(pos[2]), 6, 36))
        dropout = float(np.clip(pos[3], 0.0, 0.5))
        layers = 1 if pos[4] < 1.5 else 2
        return lr, hidden, window, dropout, layers

    positions = np.array([init_particle() for _ in range(PSOConfig.particles)], dtype=float)
    velocities = np.zeros_like(positions)

    pbest_pos = positions.copy()
    pbest_val = np.full(PSOConfig.particles, np.inf)
    gbest_pos = None
    gbest_val = np.inf

    train_len = len(train_y)
    max_valid_window = train_len - 18
    print(f"Max valid window size: {max_valid_window} (len(train)={train_len}, horizon=18)")

    for it in range(PSOConfig.iters):
        for i in range(PSOConfig.particles):
            lr, hidden, window, dropout, layers = decode(positions[i])

            if window > max_valid_window:
                val_mae = 1e12
            else:
                val_mae, state = train_one_model(train_y, window, lr, hidden, layers, dropout, device=device)

            if val_mae < pbest_val[i]:
                pbest_val[i] = val_mae
                pbest_pos[i] = positions[i].copy()
            if val_mae < gbest_val:
                gbest_val = val_mae
                gbest_pos = positions[i].copy()

        print(f"[PSO] Iter {it+1}/{PSOConfig.iters}: current global best MAE = {gbest_val:.6f}")

        r1 = rng.random(size=positions.shape)
        r2 = rng.random(size=positions.shape)
        velocities = (PSOConfig.w * velocities
                      + PSOConfig.c1 * r1 * (pbest_pos - positions)
                      + PSOConfig.c2 * r2 * (gbest_pos - positions))
        positions = positions + velocities

    best_lr, best_hidden, best_window, best_dropout, best_layers = decode(gbest_pos)
    print(f"[PSO] Best solution (val MAE={gbest_val:.6f}): "
          f"lr={best_lr:.5f}, hidden={best_hidden}, window={best_window}, "
          f"dropout={best_dropout:.3f}, layers={best_layers}")

    return {
        "best_val_mae": float(gbest_val),
        "best_hparams": {
            "lr": float(best_lr),
            "hidden": int(best_hidden),
            "window": int(best_window),
            "dropout": float(best_dropout),
            "layers": int(best_layers)
        }
    }

# -----------------------------
# Final evaluation on test set
# -----------------------------
def evaluate_on_test(best, train_y, test_y, device=None):
    """Evaluate the best model on the test set."""
    H = 18
    window = int(best["window"])
    lr = float(best["lr"])
    hidden = int(best["hidden"])
    layers = int(best["layers"])
    dropout = float(best["dropout"])

    L = len(train_y)
    max_valid_window = L - H
    if window > max_valid_window:
        print(f"[Evaluation] window={window} exceeds max={max_valid_window}, adjusting to max.")
        window = max_valid_window

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    (mu, sigma), [train_z, test_z] = zscore_fit_transform(train_y, test_y)
    ds = WindowDataset(train_z, window, H)
    if len(ds) == 0:
        raise RuntimeError(f"[Evaluation] Cannot create samples (window={window}, len(train)={L}).")

    loader = DataLoader(ds, batch_size=64, shuffle=True)
    model = RNNDirect18(hidden, layers, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for ep in range(40):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

    last_window = train_z[-window:]
    x = torch.tensor(last_window, dtype=torch.float32).view(1, window, 1).to(device)
    with torch.no_grad():
        pred = model(x).cpu().numpy().reshape(-1)
    pred_inv = zscore_inverse(pred, mu, sigma)

    test_true = np.asarray(test_y, dtype=float)
    mae = float(np.mean(np.abs(pred_inv - test_true)))
    rmse = float(np.sqrt(np.mean((pred_inv - test_true)**2)))

    print("\n=== Test Evaluation (last 18 months) ===")
    print("Best hyperparameters:", best)
    print(f"Test MAE:  {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print("True values: ", test_true.tolist())
    print("Predicted values: ", pred_inv.tolist())

    return {"MAE": mae, "RMSE": rmse, "y_true": test_true.tolist(), "y_pred": pred_inv.tolist()}

# -----------------------------
# Main workflow
# -----------------------------
def main():
    """Main workflow: data split → PSO search → validation → final testing."""
    y = np.array(RAW_SERIES, dtype=float)
    train, test = series_train_test_split(y, horizon=18)
    print(f"Data split: train={len(train)} points, test={len(test)} points (last 18 months)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # PSO search
    pso_best = pso_optimize(train, device=device)
    best = pso_best["best_hparams"]
    print("\n[PSO] Best validation MAE =", pso_best["best_val_mae"])

    # Final evaluation
    _report = evaluate_on_test(best, train, test, device=device)

if __name__ == "__main__":
    main()
