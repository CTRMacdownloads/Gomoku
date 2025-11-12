#ifndef PYTHON_AI_PROCESS_H
#define PYTHON_AI_PROCESS_H
#include "aiclient.h"
#include <QProcess>
#include <QFileInfo>
class PythonAiProcess : public AiClient {
    Q_OBJECT
public:
    explicit PythonAiProcess(const QString& pythonExe,
                             const QString& scriptPath,
                             const int msecs,
                             QObject* parent=nullptr);
    void requestMove(const std::vector<std::vector<int>>& board,
                     int player, std::optional<std::pair<int,int>> lastMove) override;
private slots:
    void onReadyRead();
    void onError(QProcess::ProcessError e);
private:
    QProcess m_proc;
    QByteArray m_buffer;
    int msecs;
};
#endif // PYTHON_AI_PROCESS_H
