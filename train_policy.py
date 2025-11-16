# train_policy.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


# ===== 1. 定义策略网络 PolicyNet =====
class PolicyNet(nn.Module):
    """
    输入: (B, 2, N, N)
      - 通道0: 当前玩家的棋子
      - 通道1: 对手的棋子
    输出: (B, N*N) 对每个格子的 logit(打分)
    """
    def __init__(self, N):
        super().__init__()
        self.N = N
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.head  = nn.Conv2d(64, 1,  kernel_size=1)

    def forward(self, x):
        # x: (B, 2, N, N)
        x = F.relu(self.conv1(x))    # (B, 32, N, N)
        x = F.relu(self.conv2(x))    # (B, 64, N, N)
        x = F.relu(self.conv3(x))    # (B, 64, N, N)
        x = self.head(x)             # (B, 1, N, N)
        x = x.view(x.size(0), -1)    # 展平为 (B, N*N)
        return x


# ===== 2. 训练函数 =====
def train_policy(
    dataset_path="gomoku_dataset.pt",
    out_model_path="gomoku_policy.pt",
    batch_size=128,
    num_epochs=20,
    lr=1e-3
):
    # ---- 2.1 读取数据集 ----
    print(f"加载数据集: {dataset_path}")
    data = torch.load(dataset_path)
    X = data["X"]      # (M, 2, N, N) float32
    y = data["y"]      # (M,) long, 每步的 action_index = r*N + c
    N = data["N"]      # 棋盘大小，例如 15

    print(f"样本数: {X.shape[0]}, 棋盘大小 N = {N}")

    # 简单划分 train / val (90% / 10%)
    num_samples = X.shape[0]
    split = int(num_samples * 0.9)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset   = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ---- 2.2 准备模型 / 优化器 / 损失函数 ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model = PolicyNet(N).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()   # 用 A* 老师的动作作为监督标签

    # ---- 2.3 训练循环 ----
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)   # (B, 2, N, N)
            yb = yb.to(device)   # (B,)

            optimizer.zero_grad()
            logits = model(xb)   # (B, N*N)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        avg_train_loss = total_loss / len(train_dataset)

        # ---- 2.4 在验证集上评估一下 ----
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)              # (B, N*N)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)

                # 计算 top-1 准确率
                preds = torch.argmax(logits, dim=1)  # (B,)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        avg_val_loss = val_loss / max(1, len(val_dataset))
        val_acc = correct / max(1, total)

        print(f"[Epoch {epoch}/{num_epochs}] "
              f"train_loss={avg_train_loss:.4f}  "
              f"val_loss={avg_val_loss:.4f}  "
              f"val_acc={val_acc:.4f}")

    # ---- 2.5 训练完成，保存权重 ----
    torch.save(model.state_dict(), out_model_path)
    print(f"训练完成，模型已保存到: {out_model_path}")


# ===== 3. 允许命令行直接运行 =====
if __name__ == "__main__":
    # 默认参数你可以根据实际情况调
    train_policy(
        dataset_path="gomoku_dataset.pt",   # 先用 generate_data.py 生成
        out_model_path="gomoku_policy.pt",  # 对弈时会在 ai_stdin.py 里加载它
        batch_size=128,
        num_epochs=20,
        lr=1e-3,
    )
