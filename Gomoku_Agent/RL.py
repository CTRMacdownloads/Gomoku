import sys, json, time, math, random
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# ================= 配置区域 =================
# "argmax": 总是选分数最高的 (稳定，但死板)
# "softmax": 按概率随机选 (灵活，每局不同)
POLICY_SAMPLING_MODE = "softmax" 
# ===========================================

POLICY_MODEL = None

class PolicyNet(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.N = N
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.head  = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.head(x)
        return x.squeeze(1)

def load_policy_model(N):
    global POLICY_MODEL
    if POLICY_MODEL is None:
        model = PolicyNet(N)
        
        # --- 使用绝对路径加载模型，防止 C++ 调用时找不到文件 ---
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "gomoku_policy.pt")
        # ------------------------------------------------
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")

        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        POLICY_MODEL = model
    return POLICY_MODEL

EMPTY, BLACK, WHITE = 0, 1, 2

def board_to_tensor(board, who):
    N = len(board)
    me_plane  = [[0.0]*N for _ in range(N)]
    opp_plane = [[0.0]*N for _ in range(N)]
    OPP = BLACK if who == WHITE else WHITE

    for r in range(N):
        for c in range(N):
            v = board[r][c]
            if v == who:
                me_plane[r][c] = 1.0
            elif v == OPP:
                opp_plane[r][c] = 1.0

    x = torch.tensor([[me_plane, opp_plane]], dtype=torch.float32)
    return x

# ================== 辅助函数 (Zobrist/规则/Minimax兜底) ==================
# 为了保持代码完整性，保留这些辅助函数，防止 RL 模型出错时无路可走

def _zobrist_init(N):
    rnd = random.Random(2025)
    return [[[rnd.getrandbits(64) for _ in range(3)] for _ in range(N)] for _ in range(N)]
ZOBRIST = None

def zobrist_hash(board):
    h = 0
    N = len(board)
    for r in range(N):
        for c in range(N):
            if board[r][c] != 0: h ^= ZOBRIST[r][c][board[r][c]]
    return h

def inb(N,r,c): return 0 <= r < N and 0 <= c < N
DIRS = [(0,1),(1,0),(1,1),(1,-1)]

def five_in_a_row(board, r, c, who):
    N = len(board)
    for dr,dc in DIRS:
        cnt = 1
        rr,cc = r+dr,c+dc
        while inb(N,rr,cc) and board[rr][cc]==who: cnt+=1; rr+=dr; cc+=dc
        rr,cc = r-dr,c-dc
        while inb(N,rr,cc) and board[rr][cc]==who: cnt+=1; rr-=dr; cc-=dc
        if cnt >= 5: return True
    return False

def legal_moves(board, radius=2):
    N = len(board)
    stones = [(r,c) for r in range(N) for c in range(N) if board[r][c]!=0]
    if not stones: return [(N//2, N//2)]
    seen=set()
    for r0,c0 in stones:
        for dr in range(-radius, radius+1):
            for dc in range(-radius, radius+1):
                r,c = r0+dr, c0+dc
                if inb(N,r,c) and board[r][c]==0: seen.add((r,c))
    return list(seen)

def can_win_points(board, who):
    res=[]
    for (r,c) in legal_moves(board):
        board[r][c]=who
        if five_in_a_row(board,r,c,who): res.append((r,c))
        board[r][c]=0
    return res

TTEntry = defaultdict(lambda: None)

# ================== 核心决策逻辑 ==================

def choose_move_with_rl(board, who, mode=None):
    """
    用策略网络选择一步落子。
    包含逻辑：如果是第一手（空棋盘），强制在天元周围 3x3 随机落子。
    """
    N = len(board)

    # 1. 【关键逻辑】检测是否是第一手
    stone_count = sum(1 for r in range(N) for c in range(N) if board[r][c] != 0)
    
    if stone_count == 0:
        center = N // 2
        # 生成中心 3x3 区域的所有点
        candidates = [
            (center + dr, center + dc)
            for dr in [-1, 0, 1]
            for dc in [-1, 0, 1]
        ]
        # 随机选一个返回
        return random.choice(candidates)

    if mode is None:
        mode = POLICY_SAMPLING_MODE

    # 2. 加载模型并推理
    model = load_policy_model(N)
    x = board_to_tensor(board, who)
    
    with torch.no_grad():
        logits = model(x)[0]

    # 3. 屏蔽非法位置
    for r in range(N):
        for c in range(N):
            if board[r][c] != EMPTY:
                logits[r, c] = -1e9

    flat = logits.view(-1)
    
    # 兜底：如果没地方下了
    if torch.all(flat <= -1e8):
        return (0, 0)

    # 4. 根据模式选择
    if mode == "softmax":
        probs = torch.softmax(flat, dim=0)
        if torch.isnan(probs).any() or probs.sum().item() <= 0:
            idx = int(torch.argmax(flat).item())
        else:
            idx = int(torch.multinomial(probs, num_samples=1).item())
    else:
        idx = int(torch.argmax(flat).item())

    r = idx // N
    c = idx % N
    return (r, c)

# 简单的 Minimax 兜底，防止 RL 出错时程序挂掉
def find_best_move_fallback(board, who):
    # 这里只做一个极简的随机合法步，作为最后的防线
    moves = legal_moves(board)
    if not moves: return (0,0)
    return random.choice(moves)

def main_loop():
    global ZOBRIST
    for line in sys.stdin:
        s = line.strip()
        if not s: continue
        try:
            req = json.loads(s)
            board = req["board"]
            player = req.get("player", "white")
            N = len(board)
            
            if ZOBRIST is None or len(ZOBRIST) != N:
                globals()["ZOBRIST"] = _zobrist_init(N)
                TTEntry.clear()
                
            who = WHITE if player == "white" else BLACK
            
            try:
                # 尝试用 RL 思考
                r, c = choose_move_with_rl(board, who)
            except Exception as e:
                # 如果 RL 失败（比如模型没加载成功），用兜底算法
                sys.stderr.write(f"RL Error: {e}\n")
                r, c = find_best_move_fallback(board, who)
                
            sys.stdout.write(json.dumps({"row": int(r), "col": int(c)}) + "\n")
            sys.stdout.flush()
            
        except Exception as e:
            sys.stdout.write(json.dumps({"row": -1, "col": -1, "error": str(e)}) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    # ================== 初始化随机种子 ==================
    # 使用当前时间戳，确保每次运行的随机性都不同
    seed_val = int(time.time() * 1000)
    random.seed(seed_val)
    torch.manual_seed(seed_val)
    # ==================================================
    
    main_loop()