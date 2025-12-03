# human-ai_generate.py
# 把 ai_stdin.py 在线对弈时记录的 ai_online_log.jsonl
# 还原出完整的人机对战棋谱（人+AI），生成 human_dataset.pt

import json
import copy
import torch

from generate_data import board_to_tensor
from ai_stdin import EMPTY, BLACK, WHITE


def count_stones(board):
    """统计棋盘上非 EMPTY 的子数，用来判断是否新开一盘。"""
    return sum(1 for row in board for v in row if v != EMPTY)


def find_single_diff(prev_board, new_board):
    """
    找出 new_board 相比 prev_board 多出来的那个子的位置 (r,c)。
    假设每一步只下一子，所以只会有一个位置不同。
    """
    N = len(prev_board)
    diff_pos = None
    for r in range(N):
        for c in range(N):
            if prev_board[r][c] != new_board[r][c]:
                if diff_pos is not None:
                    # 原来这里直接 raise，现在保留，外面用 try/except 兜底
                    raise ValueError("板面变化不止一个格子，无法确定落子位置")
                diff_pos = (r, c)
    return diff_pos


def build_dataset(log_path="ai_online_log.jsonl",
                  out_path="human_dataset.pt",
                  include_ai_moves=True):
    """
    从 ai_online_log.jsonl 中还原整盘棋：
      - 人类落子：根据 consecutive board 的差异推出来
      - AI 落子：日志里直接给 row/col

    最后生成一个 (X, y, N) 的数据集：
      X: (M, 2, N, N)
      y: (M,)
    """
    # 读入所有日志记录（按时间顺序）
    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records.append(rec)

    if not records:
        print("日志中没有记录，退出。")
        return

    all_states = []
    all_actions = []

    current_board = None   # 我们自己维护的“上一手 AI 落完后的棋盘”
    N = None
    game_idx = 0

    for idx, rec in enumerate(records):
        board_ai_before = rec["board"]           # 这一步 AI 思考前的棋盘（人刚下完）
        player_str      = rec.get("player", "white")
        ai_color        = WHITE if player_str == "white" else BLACK
        human_color     = BLACK if ai_color == WHITE else WHITE
        r_ai            = int(rec["row"])
        c_ai            = int(rec["col"])

        if N is None:
            N = len(board_ai_before)

        stones_now = count_stones(board_ai_before)

        # 情况 1：新的一盘（current_board 为空，或棋子数变少了）
        if current_board is None or stones_now < count_stones(current_board):
            game_idx += 1
            print(f"[Game {game_idx}] 棋盘重置，检测到新的一盘对局")

            # 新局开始时，假设初始棋盘是全空
            empty_board = [[EMPTY] * N for _ in range(N)]

            # 1) 如果第一条记录里棋盘已经有子，说明有人已经走了一步或多步
            if stones_now > 0:
                try:
                    hr, hc = find_single_diff(empty_board, board_ai_before)
                    first_stone = board_ai_before[hr][hc]

                    # 如果第一步是人类下的，就记一条人类样本
                    if first_stone == human_color:
                        x_human = board_to_tensor(empty_board, human_color)
                        all_states.append(x_human)
                        all_actions.append(hr * N + hc)
                    # 如果第一步就是 AI 下的（AI 先手），也可以按需记 AI 样本
                    elif first_stone == ai_color and include_ai_moves:
                        x_ai0 = board_to_tensor(empty_board, ai_color)
                        all_states.append(x_ai0)
                        all_actions.append(hr * N + hc)

                except ValueError:
                    # 说明这一局在我们开始记 log 之前已经下了好几步
                    # → 当成“从中局开始采样”，跳过人类起手，只记 AI 样本
                    print(f"警告：Game {game_idx} 的第一条记录棋子数 > 1，"
                          f"从中局开始，只记录 AI 落子，不记录这一局的起手。")

            # 2) 再记本次日志里的 AI 落子（AI 在 board_ai_before 上走 (r_ai, c_ai)）
            if include_ai_moves:
                x_ai = board_to_tensor(board_ai_before, ai_color)
                all_states.append(x_ai)
                all_actions.append(r_ai * N + c_ai)

            # 3) 更新 current_board = AI 落完之后的棋盘
            current_board = copy.deepcopy(board_ai_before)
            current_board[r_ai][c_ai] = ai_color
            continue

        # 情况 2：同一盘对局继续
        # 这一步的 board_ai_before = 人类刚走完子之后的棋盘
        # 我们的 current_board = 上一条记录 AI 落完子之后的棋盘

        try:
            hr, hc = find_single_diff(current_board, board_ai_before)
            human_stone = board_ai_before[hr][hc]

            if human_stone == human_color:
                x_human = board_to_tensor(current_board, human_color)
                all_states.append(x_human)
                all_actions.append(hr * N + hc)
            else:
                print(f"警告：第 {idx} 条记录，人类推断颜色和棋盘不符，跳过该人类样本")

        except ValueError:
            # 同一盘里某一步的棋盘变化不止 1 个格子：可能是我们开始记 log 时已经是中局
            print(f"警告：第 {idx} 条记录棋盘变化不止一个格子，"
                  f"无法确定人类落子位置，跳过该人类样本，仅记录 AI 样本。")

        # 无论人类样本是否成功推出来，AI 样本都是可靠的
        if include_ai_moves:
            x_ai = board_to_tensor(board_ai_before, ai_color)
            all_states.append(x_ai)
            all_actions.append(r_ai * N + c_ai)

        # 更新 current_board = AI 落完之后的棋盘
        current_board = copy.deepcopy(board_ai_before)
        current_board[r_ai][c_ai] = ai_color

    # -------- 打包保存 --------
    if not all_states:
        print("没有成功构建任何样本，不生成数据集。")
        return

    X = torch.stack(all_states, dim=0)             # (M, 2, N, N)
    y = torch.tensor(all_actions, dtype=torch.long)

    torch.save({"X": X, "y": y, "N": N}, out_path)
    print(f"构建完成: {out_path}, 样本数 = {X.shape[0]}, 棋盘大小 N = {N}")


if __name__ == "__main__":
    build_dataset()
