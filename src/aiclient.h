#ifndef AICLIENT_H
#define AICLIENT_H
#include <QObject>
#include <vector>
#include <optional>
class AiClient : public QObject {
    Q_OBJECT
public:
    explicit AiClient(QObject* parent=nullptr) : QObject(parent) {}
    virtual ~AiClient() = default;
    virtual void requestMove(const std::vector<std::vector<int>>& board,
                             int player, std::optional<std::pair<int,int>> lastMove) = 0;
signals:
    void moveReady(int row, int col);
    void errorOccured(const QString& msg);
};
#endif // AICLIENT_H
