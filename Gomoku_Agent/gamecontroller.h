
#ifndef GAMECONTROLLER_H
#define GAMECONTROLLER_H

#include <QObject>
#include <QString>
#include <vector>

#include "gamemodel.h"
#include "aiclient.h"

// 前向声明，避免头文件互相 include
class BoardWidget;

// 0/1/2 要和 Python 里的 EMPTY/BLACK/WHITE 对齐
enum PlayerStone {
    StoneEmpty = 0,
    StoneBlack = 1,
    StoneWhite = 2
};

// 每一方是人、A* AI 还是 RL AI
enum class PlayerKind {
    Human,
    AStarAI,
    RLAI
};

class GameController : public QObject
{
    Q_OBJECT
public:
    // 旧 UI 用的构造函数：MainWindow 里 new GameController(m_model, m_board, this)
    explicit GameController(GameModel *model,
                            BoardWidget *board,
                            QObject *parent = nullptr);

    // 新的：显式传入两种 AI（A* 和 RL）
    explicit GameController(GameModel *model,
                            AiClient *astarClient,
                            AiClient *rlClient,
                            QObject *parent = nullptr);

    // 配置对局模式
    void startNewGame(PlayerKind blackKind, PlayerKind whiteKind);
    void startAiVsAiBattle();

public slots:
    // 棋盘点击（UI -> Controller）
    void handleHumanClick(int row, int col);

    // 兼容旧 UI 的接口：菜单里点 “New Game”
    void newGame(int size);

    // 兼容旧 UI 的接口：撤销（这里先做成空操作，只发消息）
    void undo();

private slots:
    // AI 下好一步（AiClient -> Controller）
    void onAiMoveReady(int row, int col);
    void onAiErrorOccured(const QString &message);

signals:
    // 通知棋盘刷新
    void boardUpdated();

    // 通知游戏结束：0=平局, 1=黑胜, 2=白胜
    void gameFinished(int winner);

    // 兼容旧 UI：在状态栏显示一条消息
    void messageChanged(const QString &message);

private:
    // 把当前局面导出成 std::vector，方便给 Python
    std::vector<std::vector<int>> exportBoard() const;

    // 当前行棋方（1=黑, 2=白）
    int currentPlayerStone() const;

    // 依据最后一步判断胜负
    int checkGameStatus(int lastRow, int lastCol) const;

    // 当前这一方是 Human / A* / RL
    PlayerKind kindForStone(int stone) const;

    // 当前这一方对应哪个 AiClient*
    AiClient* aiForStone(int stone) const;

    // 如果轮到 AI，就请求它思考
    void requestAiMoveIfNeeded();

private:
    GameModel *m_model;      // 棋盘 + 规则
    AiClient  *m_astar;      // A* Python AI
    AiClient  *m_rl;         // RL Python AI

    PlayerKind m_blackKind;
    PlayerKind m_whiteKind;

    int  m_boardSize;
    int  m_currentStone;     // 1 = 黑, 2 = 白
    bool m_gameOver;
};

#endif // GAMECONTROLLER_H
