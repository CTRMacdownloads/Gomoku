#ifndef GAMECONTROLLER_H
#define GAMECONTROLLER_H
#include <QObject>
#include <optional>
#include <vector>
#include <QTimer>
class GameModel;
class BoardWidget;
class AiClient;
class GameController : public QObject {
    Q_OBJECT
public:
    explicit GameController(GameModel *model, BoardWidget *board, QObject *parent = nullptr);
    void setAi(AiClient* ai);
public slots:
    void handleCellClicked(int r, int c);
    void newGame(int size = 15);
    void undo();
signals:
    void messageChanged(const QString &msg);
private slots:
    void onAiMove(int r, int c);
    void onAiError(const QString& msg);
private:
    void updateStatus();
    void maybeAskAi();
private:
    GameModel *m_model;
    BoardWidget *m_board;
    AiClient *m_ai = nullptr;
    bool m_waitingAi = false;
    QTimer m_aiTimeout;
    int msecs;
};
#endif // GAMECONTROLLER_H
