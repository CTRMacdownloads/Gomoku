#include "gamemodel.h"
#include <algorithm>
GameModel::GameModel(int size) {
    reset(size);
}

void GameModel::reset(int size) {
    m_size = std::max(5, size);
    m_board.assign(m_size, std::vector<Stone>(m_size, Empty));
    m_current = Black;
    m_winner.reset();
    m_history.clear();
    m_lastMove.reset();
}

GameModel::Stone GameModel::stoneAt(int r, int c) const {
    if (!inBounds(r, c)) {
        return Empty;
    }
    return m_board[r][c];
}

bool GameModel::placeStone(int r, int c) {
    if (hasWinner()) {
        return false;
    }
    if (!inBounds(r, c)) {
        return false;
    }
    if (m_board[r][c] != Empty) {
        return false;
    }
    m_board[r][c] = m_current;
    m_history.emplace_back(r, c);
    m_lastMove = std::make_pair(r, c);
    if (checkWin(r, c, m_current)) {
        m_winner = m_current;
    } else {
        m_current = (m_current == Black ? White : Black);
    }
    return true;
}

bool GameModel::undo() {
    if (m_history.empty() || hasWinner()) {
        if (hasWinner()) {
            m_winner.reset();
        } else {
            return false;
        }
    }
    auto [r, c] = m_history.back();
    m_history.pop_back();
    m_board[r][c] = Empty;
    m_current = (m_current == Black ? White : Black);
    m_lastMove = m_history.empty() ? std::optional<std::pair<int,int>>()
                                   : std::make_optional(m_history.back());
    return true;
}

bool GameModel::inBounds(int r, int c) const {
    return r >= 0 && r < m_size && c >= 0 && c < m_size;
}

int GameModel::countDir(int r, int c, int dr, int dc, Stone who) const {
    int cnt = 0;
    int rr = r + dr, cc = c + dc;
    while (inBounds(rr, cc) && m_board[rr][cc] == who) {
        ++cnt; rr += dr; cc += dc;
    }
    return cnt;
}

bool GameModel::checkWin(int r, int c, Stone who) const {
    static const int dirs[4][2] = { {0,1}, {1,0}, {1,1}, {1,-1} };
    for (auto &d : dirs) {
        int total = 1 + countDir(r, c, d[0], d[1], who)
                      + countDir(r, c, -d[0], -d[1], who);
        if (total >= 5) {
            return true;
        }
    }
    return false;
}
