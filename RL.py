import sys, json, time, math, random
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class PolicyNet(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.N = N
        # 输入通道数 = 2（我方 / 对方）
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # 输出 1 个通道的 "logits map"
        self.head  = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # x: (B, 2, N, N)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.head(x)  # (B,1,N,N)
        return x.squeeze(1)  # -> (B,N,N)
    
    POLICY_MODEL = None

def load_policy_model(N):
    """
    懒加载策略网络，只初始化和加载一次权重。
    """
    global POLICY_MODEL
    if POLICY_MODEL is None:
        model = PolicyNet(N)
        state = torch.load("gomoku_policy.pt", map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        POLICY_MODEL = model
    return POLICY_MODEL

EMPTY, BLACK, WHITE = 0, 1, 2

def board_to_tensor(board, who):
    """
    把当前局面编码为 (1, 2, N, N) 的 float32 tensor：
    通道 0：我方棋子；通道 1：对方棋子
    """
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

    x = torch.tensor([[me_plane, opp_plane]], dtype=torch.float32)  # (1,2,N,N)
    return x

def _zobrist_init(N):
    rnd = random.Random(2025)
    table = [[[rnd.getrandbits(64) for _ in range(3)] for _ in range(N)] for _ in range(N)]
    return table
ZOBRIST = None

def zobrist_hash(board):
    h = 0
    N = len(board)
    for r in range(N):
        for c in range(N):
            v = board[r][c]
            if v != EMPTY:
                h ^= ZOBRIST[r][c][v]
    return h

def inb(N,r,c): return 0 <= r < N and 0 <= c < N

DIRS = [(0,1),(1,0),(1,1),(1,-1)]

def five_in_a_row(board, r, c, who):
    N = len(board)
    for dr,dc in DIRS:
        cnt = 1
        rr,cc = r+dr,c+dc
        while inb(N,rr,cc) and board[rr][cc]==who:
            cnt+=1; rr+=dr; cc+=dc
        rr,cc = r-dr,c-dc
        while inb(N,rr,cc) and board[rr][cc]==who:
            cnt+=1; rr-=dr; cc-=dc
        if cnt >= 5: return True
    return False

def legal_moves(board, radius=2):
    N = len(board)
    stones = [(r,c) for r in range(N) for c in range(N) if board[r][c]!=EMPTY]
    if not stones:
        m=N//2
        cand = [(m,m)]
        if N>=7:
            cand += [(m-1,m),(m,m-1),(m+1,m),(m,m+1)]
        return cand
    seen=set()
    for r0,c0 in stones:
        for dr in range(-radius, radius+1):
            for dc in range(-radius, radius+1):
                r,c = r0+dr, c0+dc
                if inb(N,r,c) and board[r][c]==EMPTY:
                    seen.add((r,c))
    return list(seen)

def all_lines_with_coords(board):
    N = len(board)
    for r in range(N):
        yield board[r][:], [(r,c) for c in range(N)]
    for c in range(N):
        col = [board[r][c] for r in range(N)]
        yield col, [(r,c) for r in range(N)]
    for s in range(-(N-1), N):
        vals, coords = [], []
        for r in range(N):
            c = r - s
            if 0 <= c < N:
                vals.append(board[r][c]); coords.append((r,c))
        if len(vals) >= 5: yield vals, coords
    for s in range(0, 2*N-1):
        vals, coords = [], []
        for r in range(N):
            c = s - r
            if 0 <= c < N:
                vals.append(board[r][c]); coords.append((r,c))
        if len(vals) >= 5: yield vals, coords

def can_win_points(board, who):
    res=[]
    for (r,c) in legal_moves(board):
        board[r][c]=who
        if five_in_a_row(board,r,c,who):
            res.append((r,c))
        board[r][c]=EMPTY
    return res

def blocking_cells_for_open_fours(board, who):
    return can_win_points(board, who)

def blocking_cells_for_open_threes(board, who):
    blocks=set()
    for vals, coords in all_lines_with_coords(board):
        n=len(vals); i=0
        while i<=n-5:
            win = vals[i:i+5]
            if (win[0]==EMPTY and win[1]==who and win[2]==who and win[3]==who and win[4]==EMPTY):
                blocks.add(coords[i]); blocks.add(coords[i+4])
            i+=1
    return list(blocks)

def is_strong_threat_after(board, r, c, who):
    board[r][c]=who
    wins = can_win_points(board, who)
    if wins:
        board[r][c]=EMPTY; return True
    o3 = blocking_cells_for_open_threes(board, who)
    board[r][c]=EMPTY
    return len(o3) >= 3

# ---------------- Heuristic ----------------
# 模式分（可按口味调整）
SCORES = {
    "FIVE":     1000000,
    "OPEN4":     50000,
    "CLOSED4":   12000,
    "OPEN3":      2500,
    "SLEEP3":      500,
    "OPEN2":       200,
    "SLEEP2":        40,
}

def line_score(vals, who):
    opp = BLACK if who==WHITE else WHITE
    n=len(vals); s=0

    # 快速五连
    cnt=0
    for x in vals:
        cnt = cnt+1 if x==who else 0
        if cnt>=5: s += SCORES["FIVE"]

    def window(k):
        for i in range(n-k+1):
            yield i, vals[i:i+k]

    for i,w in window(5):
        if w.count(who)==4 and w.count(EMPTY)==1:
            s += SCORES["CLOSED4"]
            if w[0]==EMPTY and w[-1]==EMPTY:
                s += (SCORES["OPEN4"] - SCORES["CLOSED4"])

    for i,w in window(6):
        if w.count(who)==3 and w.count(EMPTY)==3:
            if w.count(opp)==0:
                s += SCORES["OPEN3"]
    for i,w in window(5):
        if w.count(who)==2 and w.count(EMPTY)==3 and w.count(opp)==0:
            s += SCORES["OPEN2"]
    return s

def heuristic(board, who):
    me = opp = 0
    OPP = BLACK if who==WHITE else WHITE
    for vals,_ in all_lines_with_coords(board):
        me  += line_score(vals, who)
        opp += line_score(vals, OPP)
    return me - opp

TTEntry = defaultdict(lambda: None)
TT_EXACT, TT_LOWER, TT_UPPER = 0, 1, 2

def order_moves(board, moves, who):
    if not moves: return moves
    N=len(board); OPP = BLACK if who==WHITE else WHITE

    wins=set(can_win_points(board, who))
    must_block=set(can_win_points(board, OPP))
    o4 = set(blocking_cells_for_open_fours(board, who))
    o3 = set(blocking_cells_for_open_threes(board, who))

    def center_bias(r,c):
        return -((r - N/2)**2 + (c - N/2)**2)

    def density(r,c):
        d=0
        for dr in (-2,-1,0,1,2):
            for dc in (-2,-1,0,1,2):
                rr,cc=r+dr,c+dc
                if inb(N,rr,cc) and board[rr][cc]!=EMPTY: d+=1
        return d

    def key(m):
        r,c=m
        score_tuple = (
            5 if m in wins else
            4 if m in must_block else
            3 if m in o4 else
            2 if m in o3 else
            1
        )
        return (score_tuple, density(r,c), center_bias(r,c))

    return sorted(moves, key=key, reverse=True)

def minimax(board, depth, alpha, beta, who, me, t_end, tt_on=True, q_extend=True):
    now=time.time()
    if now >= t_end:
        raise TimeoutError

    h = zobrist_hash(board) if tt_on else None
    if tt_on and TTEntry[h] is not None:
        stored = TTEntry[h]
        sdepth, sval, sflag, sbest = stored
        if sdepth >= depth:
            if sflag == TT_EXACT: return sval, sbest
            if sflag == TT_LOWER and sval > alpha: alpha = sval
            elif sflag == TT_UPPER and sval < beta:  beta  = sval
            if alpha >= beta: return sval, sbest

    if depth == 0:
        val = heuristic(board, me)
        return val, None

    N=len(board); OPP = BLACK if who==WHITE else WHITE

    wins = can_win_points(board, who)
    if wins:
        return 900000 + depth, wins[0]

    must_block = can_win_points(board, OPP)
    if must_block:
        moves = must_block
    else:
        moves = legal_moves(board, radius=2)
        moves = order_moves(board, moves, who)

    if q_extend and depth==1:
        ext=[]
        for (r,c) in moves[:20]:
            if is_strong_threat_after(board, r, c, who):
                ext.append((r,c))
        if ext:
            moves = ext + moves

    best_move=None
    if who == me:
        v = -math.inf
        for (r,c) in moves:
            board[r][c]=who
            score,_ = minimax(board, depth-1, alpha, beta, OPP, me, t_end, tt_on, q_extend)
            board[r][c]=EMPTY
            if score > v:
                v = score; best_move=(r,c)
            alpha = max(alpha, v)
            if alpha >= beta:
                break
        flag = TT_EXACT
    else:
        v = math.inf
        for (r,c) in moves:
            board[r][c]=who
            score,_ = minimax(board, depth-1, alpha, beta, OPP, me, t_end, tt_on, q_extend)
            board[r][c]=EMPTY
            if score < v:
                v = score; best_move=(r,c)
            beta = min(beta, v)
            if alpha >= beta:
                break
        flag = TT_EXACT

    if tt_on and h is not None:
        TTEntry[h] = (depth, v, flag, best_move)
    return v, best_move

def opponent_threat_cells(board, opp):
    N = len(board)
    cells = []
    for r in range(N):
        for c in range(N):
            if board[r][c] != 0:
                continue
            board[r][c] = opp
            wins = can_win_points(board, opp)
            board[r][c] = 0
            if wins:
                cells.append((r, c))
    return cells

def pick_best_block(board, who, blocks):
    best, best_score = None, -10**18
    for (r,c) in blocks:
        if board[r][c] != 0:
            continue
        board[r][c] = who
        sc = heuristic(board, who)
        board[r][c] = 0
        if sc > best_score:
            best_score, best = sc, (r, c)
    return best

# ===== 采样模式开关 =====
# 可选: "argmax" 或 "softmax"
POLICY_SAMPLING_MODE = "argmax"   # 默认贪心策略；想要随机一点就改成 "softmax"


def choose_move_with_rl(board, who, mode=None):
    """
    用策略网络选择一步落子：
    - mode="argmax"  : 总是选得分最高的格子（更稳定）
    - mode="softmax" : 按概率随机采样（更有随机性）
    如果 mode=None，就用全局的 POLICY_SAMPLING_MODE。
    """

    N = len(board)
    if mode is None:
        mode = POLICY_SAMPLING_MODE

    # 1. 拿模型 & 把棋盘编码成 tensor
    model = load_policy_model(N)
    x = board_to_tensor(board, who)      # x.shape = (1, 2, N, N)

    # 2. 前向推理，得到每个格子的 score（logits）
    with torch.no_grad():
        logits = model(x)[0]             # logits.shape = (N, N)

    # 3. 屏蔽已有棋子的位置：这些地方不能下
    for r in range(N):
        for c in range(N):
            if board[r][c] != EMPTY:
                logits[r, c] = -1e9      # 相当于 -inf

    # 4. 展平为一维，按模式选择动作
    flat = logits.view(-1)

    # 如果全是 -1e9（极端情况），说明没空位，兜底随机返回一个(0,0)
    if torch.all(flat <= -1e8):
        return (0, 0)

    if mode == "softmax":
        # ---- 软最大采样模式：按概率随机选 ----
        probs = torch.softmax(flat, dim=0)   # 变成 N*N 维概率
        # 防止数值问题：如果出现 NaN 或总和为0，就退回 argmax
        if torch.isnan(probs).any() or probs.sum().item() <= 0:
            idx = int(torch.argmax(flat).item())
        else:
            idx = int(torch.multinomial(probs, num_samples=1).item())
    else:
        # ---- 默认 / 其他：贪心模式，直接选最大值 ----
        idx = int(torch.argmax(flat).item())

    r = idx // N
    c = idx % N
    return (r, c)


def find_best_move(board, who, time_limit_ms=800, max_depth=6):
    start = time.time()
    t_end = start + time_limit_ms/1000.0
    best = None

    wins = can_win_points(board, who)
    if wins:
        return wins[0]

    opp = BLACK if who==WHITE else WHITE
    must_block = can_win_points(board, opp)
    if must_block:
        return must_block[0]

    threat_blocks = opponent_threat_cells(board, opp)
    if threat_blocks:
        pick = pick_best_block(board, who, threat_blocks)
        if pick:
            return pick

    for depth in range(2, max_depth+1, 2):
        try:
            val, move = minimax(board, depth, -math.inf, math.inf, who, who, t_end, tt_on=True, q_extend=True)
            if move is not None:
                best = move
        except TimeoutError:
            break

    if best is None:
        moves = order_moves(board, legal_moves(board), who)
        best = moves[0] if moves else (0,0)
    return best

def main_loop():
    global ZOBRIST
    for line in sys.stdin:
        s = line.strip()
        if not s: continue
        try:
            req = json.loads(s)
            board = req["board"];
            player=req.get("player","white")
            N = len(board)
            if ZOBRIST is None or len(ZOBRIST)!=N:
                globals()["ZOBRIST"] = _zobrist_init(N)
                TTEntry.clear()
            who = WHITE if player=="white" else BLACK
            try:
    # 方案 B：纯 RL 出招（现在先是占位的随机版）
              r, c = choose_move_with_rl(board, who, mode=None)
            except Exception as e:
    # 如果 RL 这边出 bug，兜底用原来的 A* / minimax
              r, c = find_best_move(board, who, time_limit_ms=5000, max_depth=20)
            sys.stdout.write(json.dumps({"row":int(r),"col":int(c)})+"\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(json.dumps({"row":-1,"col":-1,"error":str(e)})+"\n")
            sys.stdout.flush()

if __name__=="__main__":
    main_loop()
