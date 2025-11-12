#ifndef GAMEMODEL_H
#define GAMEMODEL_H
#include <vector>
#include <utility>
#include <optional>
class GameModel {
public:
    enum Stone : int { Empty = 0, Black = 1, White = 2 };
    explicit GameModel(int size = 15);
    void reset(int size = 15);
    int size() const { return m_size; }
    Stone currentPlayer() const { return m_current; }
    Stone stoneAt(int r, int c) const;
    bool placeStone(int r, int c);
    bool undo();
    bool hasWinner() const { return m_winner.has_value(); }
    std::optional<Stone> winner() const { return m_winner; }
    std::optional<std::pair<int,int>> lastMove() const { return m_lastMove; }
private:
    bool inBounds(int r, int c) const;
    bool checkWin(int r, int c, Stone who) const;
    int countDir(int r, int c, int dr, int dc, Stone who) const;
private:
    int m_size;
    std::vector<std::vector<Stone>> m_board;
    Stone m_current;
    std::optional<Stone> m_winner;
    std::vector<std::pair<int,int>> m_history;
    std::optional<std::pair<int,int>> m_lastMove;
};
#endif // GAMEMODEL_H
