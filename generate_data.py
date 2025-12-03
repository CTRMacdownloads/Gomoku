# generate_data.py
import torch
from copy import deepcopy

# 从你已有的 ai_stdin.py 里把这些东西拿过来用
from ai_stdin import (
    EMPTY, BLACK, WHITE,
    find_best_move,
    legal_moves,
    five_in_a_row,
)

# ------------ 一些小工具函数 ------------

def other_side(who: int) -> int:
    return BLACK if who == WHITE else WHITE


def is_win(board, r, c, who) -> bool:
    """简单封装一下，直接用 ai_stdin 里的 five_in_a_row 判胜负。"""
    return five_in_a_row(board, r, c, who)


def board_to_tensor(board, who):
    """
    把棋盘编码成 (2, N, N) 的 tensor:
    - 通道 0: 当前玩家的棋子位置
    - 通道 1: 对手的棋子位置
    """
    N = len(board)
    x = torch.zeros(2, N, N, dtype=torch.float32)
    for r in range(N):
        for c in range(N):
            v = board[r][c]
            if v == EMPTY:
                continue
            if v == who:
                x[0, r, c] = 1.0
            else:
                x[1, r, c] = 1.0
    return x


# ------------ 1. 用 A* / minimax 自己对弈一盘 ------------

def self_play_one_game(N=15,
                       time_limit_ms=800,
                       max_depth=6):
    """
    用现有的 find_best_move 作为“老师”，让黑白双方自动对弈一整盘。
    返回：
      history: [(state_board_copy, (r,c), player), ...]
      winner:  BLACK / WHITE / None(平局)
    """
    # 初始化空棋盘
    board = [[EMPTY] * N for _ in range(N)]
    who = BLACK  # 黑棋先手
    history = []  # 保存 (state_copy, action_rc, player)

    while True:
        # 1) 复制当前局面（注意要 deepcopy）
        state = deepcopy(board)

        # 2) 老师 AI 给出当前玩家的最佳落子
        r, c = find_best_move(board, who,
                              time_limit_ms=time_limit_ms,
                              max_depth=max_depth)

        # 3) 记录样本
        history.append((state, (r, c), who))

        # 4) 真正落子
        board[r][c] = who

        # 5) 判断是否结束（胜负）
        if is_win(board, r, c, who):
            return history, who

        # 6) 判断是否平局（无合法落子）
        if not legal_moves(board):
            return history, None

        # 7) 换另一方落子
        who = other_side(who)


# ------------ 2. 连续下很多盘，生成训练数据 ------------

def generate_dataset(num_games=100,
                     N=15,
                     out_path="gomoku_dataset.pt",
                     time_limit_ms=800,
                     max_depth=6,
                     only_winner=True):
    """
    连续跑 num_games 盘自博弈，生成 (state, action) 数据集。
    - only_winner=True 时，只保留获胜方的样本（通常策略会更干净一点）
    """
    state_tensors = []
    action_indices = []

    for g in range(num_games):
        print(f"[Game {g+1}/{num_games}] self-play ...")
        history, winner = self_play_one_game(
            N=N,
            time_limit_ms=time_limit_ms,
            max_depth=max_depth
        )

        for state, (r, c), player in history:
            # 如果只想收集获胜方的落子
            if only_winner and winner is not None and player != winner:
                continue

            # 输入：从当前 player 视角编码棋盘
            x = board_to_tensor(state, player)        # (2, N, N)
            state_tensors.append(x)

            # 输出：把 (r,c) 编码成一个 index = r*N + c
            idx = r * N + c
            action_indices.append(idx)

    # 堆叠成大 tensor
    X = torch.stack(state_tensors, dim=0)           # (M, 2, N, N)
    y = torch.tensor(action_indices, dtype=torch.long)  # (M,)

    print(f"总样本数: {X.shape[0]}")
    print(f"保存到: {out_path}")
    torch.save({"X": X, "y": y, "N": N}, out_path)


# ------------ 3. 允许直接 python generate_data.py 运行 ------------

if __name__ == "__main__":
    # 你可以根据机器性能自己调这些参数
    generate_dataset(
        num_games=50,          # 想多一点就改大，比如 200、500
        N=15,
        out_path="gomoku_dataset.pt",
        time_limit_ms=600,     # 稍微少一点时间，加快自博弈
        max_depth=6,
        only_winner=True
    )
