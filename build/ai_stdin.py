import sys, json, time, math, random
from collections import defaultdict

EMPTY, BLACK, WHITE = 0, 1, 2

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

# ---------------- Utils ----------------
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
        # 首手：偏中心的几个点即可
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

# ---- 线扫描（用于启发）----
def all_lines_with_coords(board):
    N = len(board)
    # rows
    for r in range(N):
        yield board[r][:], [(r,c) for c in range(N)]
    # cols
    for c in range(N):
        col = [board[r][c] for r in range(N)]
        yield col, [(r,c) for r in range(N)]
    # diag down-right (r-c const)
    for s in range(-(N-1), N):
        vals, coords = [], []
        for r in range(N):
            c = r - s
            if 0 <= c < N:
                vals.append(board[r][c]); coords.append((r,c))
        if len(vals) >= 5: yield vals, coords
    # diag up-right (r+c const)
    for s in range(0, 2*N-1):
        vals, coords = [], []
        for r in range(N):
            c = s - r
            if 0 <= c < N:
                vals.append(board[r][c]); coords.append((r,c))
        if len(vals) >= 5: yield vals, coords

# ---------------- Threat detection ----------------
def can_win_points(board, who):
    """ who 下一手直接成五的点 """
    res=[]
    for (r,c) in legal_moves(board):
        board[r][c]=who
        if five_in_a_row(board,r,c,who):
            res.append((r,c))
        board[r][c]=EMPTY
    return res

def blocking_cells_for_open_fours(board, who):
    """
    who 的活四（任何一手可连成五）之堵点。简化：枚举空点试防一手赢。
    实际上等价于 can_win_points(who) 的集合（堵其任一点）。
    """
    return can_win_points(board, who)

def blocking_cells_for_open_threes(board, who):
    """
    who 的活三（0 who who who 0）之两个端点。
    """
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
    """落(r,c)后是否形成强威胁（活四或双活三）→ 静稳加深依据"""
    board[r][c]=who
    # 活四：对手若不应直接输
    wins = can_win_points(board, who)
    if wins:
        board[r][c]=EMPTY; return True
    # 双活三粗判：活三端点数量>=3 近似代表双三
    o3 = blocking_cells_for_open_threes(board, who)
    board[r][c]=EMPTY
    return len(o3) >= 3

# ---------------- Heuristic ----------------
# 模式分（可按口味调整）
SCORES = {
    "FIVE":     1000000,
    "OPEN4":     50000,    # 活四
    "CLOSED4":   12000,    # 冲四
    "OPEN3":      2500,
    "SLEEP3":      500,    # 眠三
    "OPEN2":       200,
    "SLEEP2":        40,
}

def line_score(vals, who):
    """
    简易模式匹配：滑窗统计 who 的若干模式（不含禁手规则）。
    """
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

    # 活四 / 冲四
    for i,w in window(5):
        # 活四：0 XXXX 0
        if w.count(who)==4 and w.count(EMPTY)==1:
            # 但窗口两边是否真“活”要看更大视野，这里用近似
            s += SCORES["CLOSED4"]
            # 判断两端是否都空（更像活四）
            if w[0]==EMPTY and w[-1]==EMPTY:
                s += (SCORES["OPEN4"] - SCORES["CLOSED4"])

    # 活三 / 眠三（用6长度观察两端气更稳）
    for i,w in window(6):
        # 形如 0 XXX 0 0 / 0 0 XXX 0 / 0 XX X 0 等近似当作活三
        if w.count(who)==3 and w.count(EMPTY)==3:
            # 不包含对手子
            if w.count(opp)==0:
                s += SCORES["OPEN3"]
    # open2/sleep2 近似
    for i,w in window(5):
        if w.count(who)==2 and w.count(EMPTY)==3 and w.count(opp)==0:
            s += SCORES["OPEN2"]
    return s

def heuristic(board, who):
    """总评估 = 我方分 - 对方分"""
    me = opp = 0
    OPP = BLACK if who==WHITE else WHITE
    for vals,_ in all_lines_with_coords(board):
        me  += line_score(vals, who)
        opp += line_score(vals, OPP)
    return me - opp

# ---------------- Search (Minimax + Alpha-Beta) ----------------
TTEntry = defaultdict(lambda: None)  # hash -> (depth, value, flag, best_move)
TT_EXACT, TT_LOWER, TT_UPPER = 0, 1, 2

def order_moves(board, moves, who):
    """威胁优先的排序：赢手/必堵/活四/活三/中心与密度偏好"""
    if not moves: return moves
    N=len(board); OPP = BLACK if who==WHITE else WHITE

    # 预打标记
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
    """返回 (value, best_move)"""
    now=time.time()
    if now >= t_end:
        raise TimeoutError

    # 置换表
    h = zobrist_hash(board) if tt_on else None
    if tt_on and TTEntry[h] is not None:
        stored = TTEntry[h]
        sdepth, sval, sflag, sbest = stored
        if sdepth >= depth:
            if sflag == TT_EXACT: return sval, sbest
            if sflag == TT_LOWER and sval > alpha: alpha = sval
            elif sflag == TT_UPPER and sval < beta:  beta  = sval
            if alpha >= beta: return sval, sbest

    # 终局 & 深度
    # 快速终局检测：最近一步是否已连五由外层控制；这里使用深度到0则回评估
    if depth == 0:
        val = heuristic(board, me)
        return val, None

    N=len(board); OPP = BLACK if who==WHITE else WHITE

    # 赢手/必堵立即裁剪
    wins = can_win_points(board, who)
    if wins:
        return 900000 + depth, wins[0]  # 越早获胜分越大

    must_block = can_win_points(board, OPP)
    # 若对手一手赢，必须堵；把这些点作为唯一分支（巨大剪枝）
    if must_block:
        moves = must_block
    else:
        moves = legal_moves(board, radius=2)
        moves = order_moves(board, moves, who)

    # 静稳扩展：若是强威胁，允许在 depth==0 时再加 1 层
    if q_extend and depth==1:
        ext=[]
        for (r,c) in moves[:20]:
            if is_strong_threat_after(board, r, c, who):
                ext.append((r,c))
        if ext:
            moves = ext + moves  # 把强威胁提前，并允许多看一步（通过上面的 depth==0 判）

    best_move=None
    if who == me:
        # Maximizer
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
        # Minimizer
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

    # 存置换表
    if tt_on and h is not None:
        TTEntry[h] = (depth, v, flag, best_move)
    return v, best_move

def opponent_threat_cells(board, opp):
    """返回所有空位 m：若对手在 m 落子，立刻会产生 can_win_points(opp) != [] 的局面。"""
    N = len(board)
    cells = []
    for r in range(N):
        for c in range(N):
            if board[r][c] != 0:
                continue
            board[r][c] = opp
            # 对手在 m 下一手后，是否出现“一步即胜”的落点（含横/竖/两条斜线，含跳三变四）
            wins = can_win_points(board, opp)
            board[r][c] = 0
            if wins:  # 这类点必须现在就堵
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

def find_best_move(board, who, time_limit_ms=800, max_depth=6):
    """
    迭代加深：逐步加深直到时间截止。
    time_limit_ms：总时限；max_depth：最高深度（偶数更稳）。
    """
    start = time.time()
    t_end = start + time_limit_ms/1000.0
    best = None

    # 0) 我方一手赢
    wins = can_win_points(board, who)
    if wins:
        return wins[0]

    # 1) 对手一手赢 → 必堵
    opp = BLACK if who==WHITE else WHITE
    must_block = can_win_points(board, opp)
    if must_block:
        return must_block[0]

    # 2) 对手一步即可制造“下一手必胜威胁”的点（覆盖斜线活三/跳三）→ 必堵
    threat_blocks = opponent_threat_cells(board, opp)
    if threat_blocks:
        pick = pick_best_block(board, who, threat_blocks)
        if pick:
            return pick

    # 迭代加深
    for depth in range(2, max_depth+1, 2):
        try:
            val, move = minimax(board, depth, -math.inf, math.inf, who, who, t_end, tt_on=True, q_extend=True)
            if move is not None:
                best = move
        except TimeoutError:
            break
    # 兜底
    if best is None:
        moves = order_moves(board, legal_moves(board), who)
        best = moves[0] if moves else (0,0)
    return best

# ---------------- I/O loop ----------------
def main_loop():
    global ZOBRIST
    for line in sys.stdin:
        s=line.strip()
        if not s: continue
        try:
            req=json.loads(s)
            board=req["board"]; player=req.get("player","white")
            N = len(board)
            if ZOBRIST is None or len(ZOBRIST)!=N:
                globals()["ZOBRIST"] = _zobrist_init(N)
                TTEntry.clear()
            who = WHITE if player=="white" else BLACK
            r,c = find_best_move(board, who, time_limit_ms=5000, max_depth=20)
            sys.stdout.write(json.dumps({"row":int(r),"col":int(c)})+"\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(json.dumps({"row":-1,"col":-1,"error":str(e)})+"\n")
            sys.stdout.flush()

if __name__=="__main__":
    main_loop()
