#include "gamecontroller.h"
#include "gamemodel.h"
#include "boardwidget.h"
#include "aiclient.h"
#include "python_ai_process.h"

GameController::GameController(GameModel *model, BoardWidget *board, QObject *parent)
    : QObject(parent), m_model(model), m_board(board) {
    msecs = 5000;
    auto *ai = new PythonAiProcess(QStringLiteral("python3"),
                                   QStringLiteral("ai_stdin.py"),
                                   msecs,
                                   this);
    setAi(ai);
    connect(board, &BoardWidget::cellClicked,
                     this, &GameController::handleCellClicked);
    m_aiTimeout.setSingleShot(true);
    connect(&m_aiTimeout, &QTimer::timeout, this, [this](){
        if (m_waitingAi) {
            m_waitingAi = false;
            emit messageChanged(QStringLiteral("AI timeout; skipping move."));
        }
    });
}

void GameController::setAi(AiClient* ai) {
    if (m_ai) QObject::disconnect(m_ai, nullptr, this, nullptr);
    m_ai = ai;
    if (!m_ai) return;
    QObject::connect(m_ai, &AiClient::moveReady, this, &GameController::onAiMove);
    QObject::connect(m_ai, &AiClient::errorOccured, this, &GameController::onAiError);
}

void GameController::handleCellClicked(int r, int c) {
    if (m_waitingAi) {
        return;
    }
    if (m_model->currentPlayer() != GameModel::Black) {
        return;
    }
    if (m_model->placeStone(r, c)) {
        m_board->update();
        updateStatus();
        maybeAskAi();
    } else {
        emit messageChanged(QStringLiteral("Invalid move."));
    }
}

void GameController::newGame(int size) {
    m_model->reset(size);
    m_board->update();
    m_waitingAi = false;
    m_aiTimeout.stop();
    updateStatus();
}

void GameController::undo() {
    if (m_waitingAi) {
        return;
    }
    if (m_model->undo()) {
        m_board->update();
        updateStatus();
    } else {
        emit messageChanged(QStringLiteral("Nothing to undo."));
    }
}

void GameController::maybeAskAi() {
    if (!m_ai) {
        return;
    }
    if (m_model->hasWinner()) {
        return;
    }
    if (m_model->currentPlayer() != GameModel::White) {
        return;
    }
    std::vector<std::vector<int>> board(m_model->size(), std::vector<int>(m_model->size(), 0));
    for (int i=0;i<m_model->size();++i) {
        for (int j=0;j<m_model->size();++j) {
            board[i][j] = static_cast<int>(m_model->stoneAt(i,j));
        }
    }
    auto lm = m_model->lastMove();
    m_waitingAi = true;
    m_aiTimeout.start(msecs);
    m_ai->requestMove(board, /*player=*/2, lm);
}

void GameController::onAiMove(int r, int c) {
    m_aiTimeout.stop();
    m_waitingAi = false;
    if (m_model->placeStone(r, c)) {
        m_board->update();
        updateStatus();
    } else {
        emit messageChanged(QStringLiteral("AI made invalid move."));
    }
}

void GameController::onAiError(const QString& msg) {
    m_aiTimeout.stop();
    m_waitingAi = false;
    emit messageChanged("AI error: " + msg);
}

void GameController::updateStatus() {
    if (m_model->hasWinner()) {
        auto w = *m_model->winner();
        QString who = (w == GameModel::Black ? "Black" : "White");
        emit messageChanged(QString("Winner: %1").arg(who));
    } else {
        auto cur = m_model->currentPlayer();
        QString who = (cur == GameModel::Black ? "Black" : "White");
        emit messageChanged(QString("Turn: %1").arg(who));
    }
}
