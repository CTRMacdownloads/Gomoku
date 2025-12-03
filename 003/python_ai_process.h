#pragma once

#include <QObject>
#include <QProcess>
#include <QVector>
#include "aiclient.h" // 引入 AiClient

class PythonAiProcess : public AiClient // 继承 AiClient
{
    Q_OBJECT
public:
    explicit PythonAiProcess(const QString &scriptPath,
                             QObject *parent = nullptr);

    // 启动 python 进程（如果还没启动）
    void start();

    // 实现 AiClient::requestMove 接口
    // board: N×N，0=EMPTY, 1=BLACK, 2=WHITE
    void requestMove(const std::vector<std::vector<int>> &board,
                     int player,
                     std::optional<std::pair<int,int>> lastMove) override;

    // *** 关键修复：不要在这里重新定义 signals ***
    // 直接使用基类 AiClient 中的 moveReady 和 errorOccured

private slots:
    void onReadyRead();
    void onProcessError(QProcess::ProcessError);
    void onFinished(int exitCode, QProcess::ExitStatus status);

private:
    QString  m_scriptPath;
    QProcess m_proc;
    QByteArray m_buffer;
    bool m_started = false;

    void sendJsonRequest(const QVector<QVector<int>> &board, int who);
    void handleOneJsonLine(const QByteArray &line);
};