#include "gamecontroller.h"
#include <QTimer> // 必须引入这个
#include "gamemodel.h"
#include "boardwidget.h"
#include "aiclient.h"
#include "python_ai_process.h"

#include <QDebug>
#include <vector>
#include <optional>

// ------------------------ 构造函数 ------------------------

GameController::GameController(GameModel *model,
                               BoardWidget *board,
                               QObject *parent)
    : QObject(parent)
    , m_model(model)
    , m_astar(nullptr)
    , m_rl(nullptr)
    , m_blackKind(PlayerKind::Human)
    , m_whiteKind(PlayerKind::Human)
    , m_boardSize(model ? model->size() : 15)
    , m_currentStone(StoneBlack)
    , m_gameOver(false)
{
    Q_UNUSED(board);
    Q_ASSERT(m_model);
}

GameController::GameController(GameModel *model,
                               AiClient *astarClient,
                               AiClient *rlClient,
                               QObject *parent)
    : QObject(parent)
    , m_model(model)
    , m_astar(astarClient)
    , m_rl(rlClient)
    , m_blackKind(PlayerKind::Human)
    , m_whiteKind(PlayerKind::Human)
    , m_boardSize(model ? model->size() : 15)
    , m_currentStone(StoneBlack)
    , m_gameOver(false)
{
    Q_ASSERT(m_model);

    if (m_astar) {
        connect(m_astar, &AiClient::moveReady,
                this, &GameController::onAiMoveReady);
        connect(m_astar, &AiClient::errorOccured,
                this, &GameController::onAiErrorOccured);
    }

    if (m_rl) {
        connect(m_rl, &AiClient::moveReady,
                this, &GameController::onAiMoveReady);
        connect(m_rl, &AiClient::errorOccured,
                this, &GameController::onAiErrorOccured);
    }
}

// ------------------------ 对局控制 ------------------------

void GameController::startNewGame(PlayerKind blackKind, PlayerKind whiteKind)
{
    m_blackKind = blackKind;
    m_whiteKind = whiteKind;

    if (m_model) {
        m_model->reset();              // 清空棋盘
        m_boardSize = m_model->size();
    }

    m_gameOver     = false;
    m_currentStone = StoneBlack;       // 黑先

    emit boardUpdated();
    emit messageChanged(QStringLiteral("New game started."));
    requestAiMoveIfNeeded();           // 如果黑方是 AI，就先让 AI 下
}

void GameController::startAiVsAiBattle()
{
    // 黑方 = A*，白方 = RL
    startNewGame(PlayerKind::AStarAI, PlayerKind::RLAI);
}

// MainWindow 菜单 “New Game” 调用
void GameController::newGame(int size)
{
    m_boardSize = size;                // 先记一下尺寸（目前 GameModel 自己也有 size()）

    // 默认：人类执黑，A* 执白
    startNewGame(PlayerKind::Human, PlayerKind::AStarAI);
}

// MainWindow 菜单 “Undo” 调用
void GameController::undo()
{
    // 如果你以后在 GameModel 里实现了 undo()，可以在这里调用。
    // 目前先给个提示，不真正撤销：
    emit messageChanged(QStringLiteral("Undo is not implemented in RL controller."));
}

// ------------------------ 人类 / AI 落子入口 ------------------------

void GameController::handleHumanClick(int row, int col)
{
    if (m_gameOver) return;

    PlayerKind kind = kindForStone(m_currentStone);
    if (kind != PlayerKind::Human) {
        // 当前轮不是人类，忽略点击
        return;
    }

    // 人类尝试落子
    if (!m_model->placeStone(row, col)) {
        qDebug() << "非法或失败的落子: (" << row << "," << col << ")";
        return;
    }

    emit boardUpdated();

    int status = checkGameStatus(row, col);
    if (status == StoneBlack || status == StoneWhite) {
        m_gameOver = true;
        emit gameFinished(status);
        return;
    } else if (status == 3) {
        m_gameOver = true;
        emit gameFinished(0);
        return;
    }

    // 换边
    m_currentStone = (m_currentStone == StoneBlack) ? StoneWhite : StoneBlack;
    requestAiMoveIfNeeded();
}

void GameController::onAiMoveReady(int row, int col)
{
    if (m_gameOver) return;

    // AI 落子逻辑 (保持不变)
    if (!m_model->placeStone(row, col)) {
        // ... (错误处理保持不变)
        return;
    }

    emit boardUpdated(); // 通知界面刷新

    // 检查胜负逻辑 (保持不变)
    int status = checkGameStatus(row, col);
    if (status != 0) { // 有胜负或平局
        m_gameOver = true;
        emit gameFinished(status == 3 ? 0 : status);
        return;
    }

    // 换边
    m_currentStone = (m_currentStone == StoneBlack) ? StoneWhite : StoneBlack;

    // =========== 修改重点在这里 ===========
    // 原来的代码是直接调用 requestAiMoveIfNeeded();
    // 现在改为用 QTimer 延时 200 毫秒再调用
    // 这样界面就有时间刷新了，看起来也更自然
    QTimer::singleShot(200, this, &GameController::requestAiMoveIfNeeded);
    // ===================================
}

void GameController::onAiErrorOccured(const QString &message)
{
    qWarning() << "AI error:" << message;
    emit messageChanged(QStringLiteral("AI error: ") + message);
}

// ------------------------ 辅助函数 ------------------------

std::vector<std::vector<int>> GameController::exportBoard() const
{
    int N = m_model ? m_model->size() : m_boardSize;
    std::vector<std::vector<int>> board(N, std::vector<int>(N, 0));

    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            auto s = m_model->stoneAt(r, c);      // GameModel::Stone 枚举
            board[r][c] = static_cast<int>(s);    // 转成 0/1/2
        }
    }
    return board;
}

int GameController::currentPlayerStone() const
{
    return m_currentStone;
}

int GameController::checkGameStatus(int /*lastRow*/, int /*lastCol*/) const
{
    if (!m_model) return 0;

    // winner() : std::optional<GameModel::Stone>
    auto w = m_model->winner();
    if (w.has_value()) {
        if (*w == GameModel::Stone::Black) {
            return StoneBlack;
        } else if (*w == GameModel::Stone::White) {
            return StoneWhite;
        }
    }

    // 暂时不判断平局（如果以后有 m_model->isDraw() 可以在这里加）
    return 0;   // 0 = 未结束
}

PlayerKind GameController::kindForStone(int stone) const
{
    return (stone == StoneBlack) ? m_blackKind : m_whiteKind;
}

AiClient *GameController::aiForStone(int stone) const
{
    PlayerKind kind = kindForStone(stone);
    switch (kind) {
    case PlayerKind::AStarAI:
        return m_astar;
    case PlayerKind::RLAI:
        return m_rl;
    default:
        return nullptr;
    }
}

void GameController::requestAiMoveIfNeeded()
{
    if (m_gameOver) return;

    PlayerKind kind = kindForStone(m_currentStone);
    if (kind == PlayerKind::Human)
        return;

    AiClient *ai = aiForStone(m_currentStone);
    if (!ai) return;

    // 1. 当前局面
    std::vector<std::vector<int>> board = exportBoard();

    // 2. 当前执子方（1=黑, 2=白）
    int who = m_currentStone;

    // 3. 上一步棋（GameModel 里维护）
    std::optional<std::pair<int,int>> last;
    if (m_model) {
        last = m_model->lastMove();
    }

    ai->requestMove(board, who, last);
}


