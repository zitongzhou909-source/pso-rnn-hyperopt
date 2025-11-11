# ============================================
# PSO + RNN 稳健版：详细中文报错定位 & 合法性检查
# 适用：M4 月度 Macro 单条序列；Direct 18 步
# ============================================

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

# -----------------------------
# 原始时间序列（你贴的第一条 M4 Macro 月度）
# -----------------------------
RAW_SERIES = [
8000,8350,8570,7700,7080,6520,6070,6650,6830,5710,5260,5470,7870,7360,8470,7880,6750,6860,6220,6650,
5450,5280,4970,5550,7750,7760,7190,7440,6590,6210,6010,6390,5780,5700,4680,4970,6850,6740,7490,6250,
6900,5870,5610,6310,6110,6090,5810,6060,6950,7820,8270,7840,7850,6710,6220,7430,6560,7080,6680,6900,
8680,8450,8670,8470,7910,8140,7310,7860,7730,7330,7430,7150,8720,8340,8840,8780,8250,8180,7570,9280,
9220,9140,8950,8540,9360,9750,9270,8690,8200,7610,7160,8280,7370,7130,6840,7260,7430,7950,7790,8090,
7480,6700,6650,6960,6410,6310,5930,5980,6730,7410,7200,6960,6780,5720,6040,5990,6210,6460,5490,5790,
6350,6230,5940,6400,6610,5840,5350,6160,6260,5760,5450,5350,6230,6520,7230,6500,6230,5960,4970,5350,
4840,4710,4670,4670,5190,5800,6120,5140,4670,4190,4430,4840,4660,4350,4390,4790,5510,5760,5780,5470,
5020,4770,4330,4330,4270,3820,3550,4390,5760,6030,6140,5830,5420,5160,4620,5420,5600,5920,5190,5640,
5700,6110,7290,7260,7040,6100,6110,7090,7070,6150,5630,6210,7250,7480,7580,7610,5970,7100,6100,7040,
7060,6270,5670,5600,7190,7590,7310,5950,6220,5680,5450,7160,6110,6940,5660,5770,8050,7450,8190,6590,
5920,4620,5320,6060,5300,5020,5010,4640,6980,6090,7000,5370,5230,5140,4570,5360,5300,4640,4740,5250,
6350,6860,6360,5660,5270,4910,5160,5620,5180,4680,3860,3890,6340,6210,6100,4980,4260,3910,3550,4680,
4120,4740,3100,3810,4710,5700,5040,4490,3930,3500,3320,4400,4190,3670,3780,4330,5430,5610,5020,4090,
3680,3290,3360,4300,4090,3110,2690,3700,6030,5450,5830,4080,3910,3840,3650,4630,2980,3550,3690,3610,
5250,5480,5400,4380,4290,4200,4190,4860,5080,4770,4470,4840,5610,6450,5960,5300,4990,4680,4460,5710,
4830,5260,4400,4780,7040,7490,7450,5560,5000,5420,5160,5110,4900,5300,4330,4680,5800,7410,6760,5680,
5610,5140,4990,5550,4720,4680,4780,4530,6810,6830,6640,5680,5020,4940,3930,4570,4720,4610,4140,4290,
5780,6590,5950,3960,4240,4000,3950,4760,4050,4190,4080,4280,5720,7040,6950,5250,4920,4820,4510,4710,
3720,4250,3730,4610,6540,7700,7210,5730,5520,5270,5110,6520,5630,5980,5750,7020,9910,11290,11180,9280,
7850,7480,7970,9480,8800,8810,7690,7710,9660,11870,12920,10890,9500,9490,9500,10880,9270,8880,7780,
8360,12320,13490,12710,10130,9240,9640,8920,8900,8480,7870,7860,8990,10790,11820,10590,9430,9070,8250,
8740,9120,8280,7860,7150,8110,10860,10730,9610,8270,9200,6660,6270,7250,6830,6810,5810,6220,7450,9370,
7980,6050,5640,6220,5740,6040,5130,5090,5210,4910,6890,8720,7790,4770,5060,4720,4450,5120,5960,6560,
4900,4520,7370,9050,7780,5380,4700,4490,4010
]

# -----------------------------
# 通用工具
# -----------------------------
def series_train_test_split(y, horizon=18):
    y = np.asarray(y, dtype=float)
    assert len(y) > horizon, "序列太短，无法切出测试集"
    return y[:-horizon], y[-horizon:]

def zscore_fit_transform(train, *others):
    mu, sigma = float(np.mean(train)), float(np.std(train))
    if sigma < 1e-12: sigma = 1.0
    def tf(x): return (np.asarray(x, dtype=float) - mu) / sigma
    outs = [tf(train)] + [tf(o) for o in others]
    return (mu, sigma), outs

def zscore_inverse(x, mu, sigma):
    return np.asarray(x)*sigma + mu

# -----------------------------
# 数据集与模型
# -----------------------------
class WindowDataset(Dataset):
    def __init__(self, y, window, horizon=18):
        self.y = np.asarray(y, dtype=float)
        self.window = int(window)
        self.horizon = int(horizon)
        self.X, self.Y = [], []
        N = len(self.y) - self.window - self.horizon + 1
        if N <= 0:
            self.X = []
            self.Y = []
        else:
            for t in range(N):
                self.X.append(self.y[t:t+self.window])
                self.Y.append(self.y[t+self.window:t+self.window+self.horizon])
        if len(self.X) == 0:
            print(f"[构造样本] 警告：无法形成样本 (window={window}, horizon={horizon}, len(y)={len(self.y)})")
        self.X = torch.tensor(self.X, dtype=torch.float32).unsqueeze(-1)  # [N, W, 1]
        self.Y = torch.tensor(self.Y, dtype=torch.float32)               # [N, 18]
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.Y[i]

class RNNDirect18(nn.Module):
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
        out, _ = self.rnn(x)        # [B, W, H]
        last = out[:, -1, :]        # [B, H]
        return self.head(last)      # [B, 18]

# -----------------------------
# 验证划分（末 20% 窗口）
# -----------------------------
def split_windows_for_val(ds, val_ratio=0.2):
    n = len(ds)
    if n == 0:
        return Subset(ds, []), Subset(ds, [])
    k = max(int(round(n*(1-val_ratio))), 1)
    idx = list(range(n))
    return Subset(ds, idx[:k]), Subset(ds, idx[k:])

# -----------------------------
# 训练一个模型（返回验证 MAE 及最佳状态）
# -----------------------------
def train_one_model(train_y, window, lr, hidden, layers, dropout, epochs=60, batch_size=64, device=None):
    H = 18
    L = len(train_y)
    max_valid_window = L - H
    if window > max_valid_window:
        print(f"[粒子无效] window={window} > 允许最大值={max_valid_window}（len(train)={L}, horizon=18）")
        return 1e12, {"reason": "window_too_large"}

    (mu, sigma), [train_z] = zscore_fit_transform(train_y)
    ds = WindowDataset(train_z, window, H)
    if len(ds) == 0:
        return 1e12, {"reason": "no_samples"}

    tr_sub, val_sub = split_windows_for_val(ds, 0.2)
    if len(val_sub) == 0:
        print("[验证集为空] 该超参数组合无法形成验证窗口，返回大罚分")
        return 1e12, {"reason": "no_val"}

    tr_loader = DataLoader(tr_sub, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False, drop_last=False)

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

        # 验证
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb.to(device)).cpu().numpy()
                val_preds.append(pred); val_trues.append(yb.numpy())
        if len(val_preds) == 0:
            print("[验证聚合] 未收集到任何验证批次，返回大罚分")
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
# PSO 搜索
# -----------------------------
class PSOConfig:
    particles = 8
    iters = 10
    w = 0.7
    c1 = 1.4
    c2 = 1.4
    seed = 42

def pso_optimize(train_y, device=None):
    rng = np.random.default_rng(PSOConfig.seed)

    def init_particle():
        # [lr_log10, hidden, window, dropout, layers]
        return np.array([
            rng.uniform(-4, -2),          # lr_log10
            rng.integers(32, 257),        # hidden
            rng.integers(6, 37),          # window
            rng.uniform(0.0, 0.5),        # dropout
            rng.integers(1, 3)            # layers in {1,2}
        ], dtype=float)

    def decode(pos):
        lr = 10**pos[0]
        hidden = int(round(pos[1])); hidden = min(max(hidden, 32), 256)
        window = int(round(pos[2])); window = min(max(window, 6), 36)
        dropout = float(pos[3]);     dropout = float(np.clip(dropout, 0.0, 0.5))
        layers = int(round(pos[4])); layers = 1 if layers < 2 else 2
        return lr, hidden, window, dropout, layers

    positions = np.array([init_particle() for _ in range(PSOConfig.particles)], dtype=float)
    velocities = np.zeros_like(positions)

    pbest_pos = positions.copy()
    pbest_val = np.full(PSOConfig.particles, np.inf)
    gbest_pos = None
    gbest_val = np.inf

    train_len = len(train_y)
    max_valid_window = train_len - 18
    print(f"可用 window 最大值：{max_valid_window}（len(train)={train_len}, horizon=18）")

    for it in range(PSOConfig.iters):
        for i in range(PSOConfig.particles):
            lr, hidden, window, dropout, layers = decode(positions[i])

            # 粗过滤：若 window 不合法，直接给超大惩罚，避免进入训练
            if window > max_valid_window:
                val_mae = 1e12
                state = {"reason": "window_exceeds_max"}
            else:
                val_mae, state = train_one_model(
                    train_y, window, lr, hidden, layers, dropout,
                    epochs=60, batch_size=64, device=device
                )

            # 更新个体/全局最优
            if val_mae < pbest_val[i]:
                pbest_val[i] = val_mae
                pbest_pos[i] = positions[i].copy()
            if val_mae < gbest_val:
                gbest_val = val_mae
                gbest_pos = positions[i].copy()

        print(f"[PSO] 第 {it+1}/{PSOConfig.iters} 轮：当前全局最优验证 MAE = {gbest_val:.6f}")

        # 速度与位置更新
        r1 = rng.random(size=positions.shape)
        r2 = rng.random(size=positions.shape)
        velocities = (PSOConfig.w * velocities
                      + PSOConfig.c1 * r1 * (pbest_pos - positions)
                      + PSOConfig.c2 * r2 * (gbest_pos - positions))
        positions = positions + velocities

    best_lr, best_hidden, best_window, best_dropout, best_layers = decode(gbest_pos)
    print(f"[PSO] 最优解（验证 MAE={gbest_val:.6f}）：",
          f"lr={best_lr:.5f}, hidden={best_hidden}, window={best_window}, ",
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
# 最终评估（测试集）
# -----------------------------
def evaluate_on_test(best, train_y, test_y, device=None):
    H = 18
    window = int(best["window"])
    lr = float(best["lr"])
    hidden = int(best["hidden"])
    layers = int(best["layers"])
    dropout = float(best["dropout"])

    L = len(train_y)
    max_valid_window = L - H
    if window > max_valid_window:
        print(f"[最终评估] window={window} 超出允许上限 {max_valid_window}，自动收缩到上限。")
        window = max_valid_window

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    (mu, sigma), [train_z, test_z] = zscore_fit_transform(train_y, test_y)
    ds = WindowDataset(train_z, window, H)
    if len(ds) == 0:
        raise RuntimeError(f"[最终评估] 无法形成训练样本（window={window}, len(train)={L}），请检查。")

    loader = DataLoader(ds, batch_size=64, shuffle=True)
    model = RNNDirect18(hidden, layers, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for ep in range(40):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); pred = model(xb)
            loss = loss_fn(pred, yb); loss.backward(); opt.step()

    # 用训练集最后一个窗口做 18 步预测
    last_window = train_z[-window:]
    x = torch.tensor(last_window, dtype=torch.float32).view(1, window, 1).to(device)
    with torch.no_grad():
        pred = model(x).cpu().numpy().reshape(-1)
    pred_inv = zscore_inverse(pred, mu, sigma)

    test_true = np.asarray(test_y, dtype=float)
    mae = float(np.mean(np.abs(pred_inv - test_true)))
    rmse = float(np.sqrt(np.mean((pred_inv - test_true)**2)))

    print("\n=== 测试集评估（最后 18 个月） ===")
    print("最优超参：", best)
    print(f"Test MAE:  {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print("真实值: ", test_true.tolist())
    print("预测值: ", pred_inv.tolist())

    return {"MAE": mae, "RMSE": rmse, "y_true": test_true.tolist(), "y_pred": pred_inv.tolist()}

# -----------------------------
# 主流程
# -----------------------------
def main():
    y = np.array(RAW_SERIES, dtype=float)
    train, test = series_train_test_split(y, horizon=18)
    print(f"数据划分：训练集 {len(train)} 点，测试集 {len(test)} 点（最后18个月）")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("设备：", device)

    # PSO 搜索
    pso_best = pso_optimize(train, device=device)
    best = pso_best["best_hparams"]
    print("\n[PSO] 验证集最优 MAE =", pso_best["best_val_mae"])

    # 最终评估
    _report = evaluate_on_test(best, train, test, device=device)

if __name__ == "__main__":
    main()


