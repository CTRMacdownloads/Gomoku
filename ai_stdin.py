import sys, json, time, math, random
from collections import defaultdict

EMPTY, BLACK, WHITE = 0, 1, 2
ZOBRIST = None
TTEntry = defaultdict(lambda: None)
TT_EXACT, TT_LOWER, TT_UPPER = 0, 1, 2

def _zobrist_init(N):
    rnd = random.Random(2025)
    table = [[[rnd.getrandbits(64) for _ in range(3)] for _ in range(N)] for _ in range(N)]
    return table

def zobrist_hash(board):
    h = 0
    N = len(board)
    for r in range(N):
        for c in range(N):
            v = board[r][c]
            if v != EMPTY:
                h ^= ZOBRIST[r][c][v]
    return h

# ---------------- Imminent Move ----------------
def inBoard(N, r, c):
    return 0 <= r < N and 0 <= c < N

DIRECTIONS = [
    (0, 1),
    (1, 0),
    (1, 1),
    (1, -1)
]

def legal_moves(board, radius = 2):
    N = len(board)
    stones = [(r, c) for r in range(N) for c in range(N) if board[r][c] != EMPTY]
    if not stones:
        return [(N // 2, N // 2)]
    candidates = set()
    for r0, c0 in stones:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                r,c = r0 + dr, c0 + dc
                if inBoard(N, r, c) and board[r][c] == EMPTY:
                    candidates.add((r, c))
    return list(candidates)

def win(board, r, c, player):
    N = len(board)
    for dr, dc in DIRECTIONS:
        cnt = 1
        rr, cc = r + dr,c + dc
        while inBoard(N, rr, cc) and board[rr][cc] == player:
            cnt += 1
            rr += dr
            cc += dc
        rr, cc = r - dr, c - dc
        while inBoard(N, rr, cc) and board[rr][cc] == player:
            cnt += 1
            rr -= dr
            cc -= dc
        if cnt >= 5:
            return True
    return False

def live_four(board, r, c, player):
    N = len(board)
    for dr, dc in DIRECTIONS:
        cnt = 1
        rr, cc = r + dr,c + dc
        while inBoard(N, rr, cc) and board[rr][cc] == player:
            cnt += 1
            rr += dr
            cc += dc
        open1 = inBoard(N, rr, cc) and board[rr][cc] == EMPTY
        rr, cc = r - dr, c - dc
        while inBoard(N, rr, cc) and board[rr][cc] == player:
            cnt += 1
            rr -= dr
            cc -= dc
        open2 = inBoard(N, rr, cc) and board[rr][cc] == EMPTY
        if cnt == 4 and open1 and open2:
            return True
    return False

def rush_four(board, r, c, player):
    N = len(board)
    for dr, dc in DIRECTIONS:
        cnt = 1
        rr, cc = r + dr,c + dc
        while inBoard(N, rr, cc) and board[rr][cc] == player:
            cnt += 1
            rr += dr
            cc += dc
        open1 = inBoard(N, rr, cc) and board[rr][cc] == EMPTY
        rr, cc = r - dr, c - dc
        while inBoard(N, rr, cc) and board[rr][cc] == player:
            cnt += 1
            rr -= dr
            cc -= dc
        open2 = inBoard(N, rr, cc) and board[rr][cc] == EMPTY
        if cnt == 4 and (open1 ^ open2):
            return True
    return False

def threat_for_direct_win(board, player):
    res = []
    for (r, c) in legal_moves(board, 1):
        board[r][c] = player
        if win(board, r, c, player):
            res.append((r, c))
        board[r][c] = EMPTY
    return res

def threat_for_live_four(board, player):
    res = []
    for (r, c) in legal_moves(board, 1):
        board[r][c] = player
        if live_four(board, r, c, player):
            res.append((r, c))
        board[r][c] = EMPTY
    return res

# ---------------- Heuristic ----------------
SCORES = {
    "FIVE":     1000000,
    "OPEN4":     50000,
    "CLOSED4":   12000,
    "OPEN3":      2500,
    "SLEEP3":      500,
    "OPEN2":       200,
    "SLEEP2":        40,
}

def heuristic(board, player):
    player_score = opp_score = 0
    opp = BLACK if player==WHITE else WHITE
    for vals in all_lines(board):
        player_score += line_score(vals, player)
        opp_score += line_score(vals, opp)
    return player_score - opp_score

def all_lines(board):
    N = len(board)
    lines = []

    # rows
    for r in range(N):
        lines.append((board[r][:]))

    # columns
    for c in range(N):
        lines.append([board[r][c] for r in range(N)])

    # main diagonals:
    for s in range(-(N - 1), N):
        vals, coords = [], []
        for r in range(N):
            c = r - s
            if 0 <= c < N:
                vals.append(board[r][c])
        if len(vals) >= 5:
            lines.append(vals)

    # anti-diagonals:
    for s in range(0, 2 * N - 1):
        vals, coords = [], []
        for r in range(N):
            c = s - r
            if 0 <= c < N:
                vals.append(board[r][c])
        if len(vals) >= 5:
            lines.append((vals, coords))

    return lines

def line_score(vals, player):
    opp = BLACK if player == WHITE else WHITE
    n = len(vals)
    s = 0
    cnt = 0
    for x in vals:
        cnt = cnt + 1 if x == player else 0
        if cnt >= 5:
            s += SCORES["FIVE"]

    def window(k):
        for i in range(n - k + 1):
            yield i, vals[i : i + k]

    for i, w in window(5):
        if w.count(player) == 4 and w.count(EMPTY) == 1:
            s += SCORES["CLOSED4"]
            if w[0] == EMPTY and w[-1] == EMPTY:
                s += SCORES["OPEN4"] - SCORES["CLOSED4"]
    for i, w in window(6):
        if w.count(player) == 3 and w.count(EMPTY) == 3:
            if w.count(opp) == 0:
                s += SCORES["OPEN3"]
    for i, w in window(5):
        if w.count(player) == 2 and w.count(EMPTY) == 3 and w.count(opp) == 0:
            s += SCORES["OPEN2"]
    return s

def ordered_moves(board, moves, player):
    if not moves:
        return []
    opp = BLACK if player == WHITE else WHITE

    # Chances for Direct Win
    wins = threat_for_direct_win(board, player)
    # Opponent with 1-step Winning Opportunity
    must_block = threat_for_direct_win(board, opp)
    # Own 1-step Threats For LIVE-FOUR
    own_live_four = threat_for_live_four(board, player)
    # Opponent 1-step Threats For LIVE-FOUR
    opp_live_four = threat_for_live_four(board, opp)

    def density(r,c):
        d = 0
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                rr, cc = r + dr, c + dc
                if inBoard(len(board), rr, cc) and board[rr][cc] != EMPTY:
                    d += 4 - abs(dr) - abs(dc)
        return d

    def key(m):
        r, c = m
        score_tuple = (
            5 if m in wins else
            4 if m in must_block else
            3 if m in own_live_four else
            2 if m in opp_live_four else
            1
        )
        return (score_tuple, density(r,c))

    return sorted(moves, key = key, reverse = True)

def minimax(board, depth, alpha, beta, current_player, player, t_end, tt_on = True):
    now = time.time()
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
        val = heuristic(board, player)
        return val, None

    opp = BLACK if current_player == WHITE else WHITE

    moves = legal_moves(board, 2)
    moves = ordered_moves(board, moves, current_player)

    best_move = None
    if current_player == player:
        v = -math.inf
        for (r,c) in moves:
            board[r][c] = current_player
            score, _ = minimax(board, depth - 1, alpha, beta, opp, player, t_end, tt_on)
            board[r][c] = EMPTY
            if score > v:
                v = score
                best_move = (r, c)
            alpha = max(alpha, v)
            if alpha >= beta:
                break
        flag = TT_EXACT
    else:
        v = math.inf
        for (r,c) in moves:
            board[r][c]=current_player
            score,_ = minimax(board, depth - 1, alpha, beta, opp, player, t_end, tt_on)
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

def find_best_move(board, player, time_limit_ms = 1000, max_depth = 10):
    # Chances for Direct Win
    wins = threat_for_direct_win(board, player)
    if wins:
        return wins[0]

    opp = BLACK if player == WHITE else WHITE

    # Opponent with 1-step Winning Opportunity
    must_block = threat_for_direct_win(board, opp)
    if must_block:
        return must_block[0]

    # Own 1-step Threats For LIVE-FOUR
    own_live_four = threat_for_live_four(board, player)
    if own_live_four:
        return own_live_four[0]

    # Opponent 1-step Threats For LIVE-FOUR
    opp_live_four = threat_for_live_four(board, opp)
    if opp_live_four:
        return opp_live_four[0]

    # A* search
    start = time.time()
    t_end = start + time_limit_ms / 1000.0
    best = None
    for depth in range(2, max_depth + 1, 2):
        try:
            val, move = minimax(board, depth, -math.inf, math.inf, player, player, t_end)
            if move is not None:
                best = move
        except TimeoutError:
            break
    if best is None:
        moves = ordered_moves(board, legal_moves(board), player)
        if moves:
            best = moves
        else:
            raise RuntimeError("Last resort for best move failed")
    return best

def main_loop():
    global ZOBRIST
    for line in sys.stdin:
        s = line.strip()
        if not s: continue
        try:
            req = json.loads(s)
            board = req["board"]
            player = req.get("player", "white")
            if ZOBRIST is None or len(ZOBRIST) != len(board):
                globals()["ZOBRIST"] = _zobrist_init(len(board))
                TTEntry.clear()
            player = WHITE if player == "white" else BLACK
            r, c = find_best_move(board, player, 5000, 20)
            sys.stdout.write(json.dumps({"row": int(r), "col": int(c)}) + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stderr.write(json.dumps({"row": -1, "col": -1, "error": str(e)}) + "\n")
            sys.stderr.flush()

if __name__=="__main__":
    main_loop()
