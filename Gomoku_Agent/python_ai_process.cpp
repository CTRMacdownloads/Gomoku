#include "python_ai_process.h"
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QCoreApplication>
#include <QDebug>
#include <QVector>
#include <vector>

// 构造函数
PythonAiProcess::PythonAiProcess(const QString &scriptPath, QObject *parent)
    : AiClient(parent), // 初始化基类 AiClient
      m_scriptPath(scriptPath)
{
    connect(&m_proc, &QProcess::readyReadStandardOutput,
            this, &PythonAiProcess::onReadyRead);
    // 连接 QProcess 自身的 errorOccurred (两个r) 到我们的槽
    connect(&m_proc, &QProcess::errorOccurred,
            this, &PythonAiProcess::onProcessError);
    connect(&m_proc, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, &PythonAiProcess::onFinished);
}

void PythonAiProcess::start()
{
    if (m_started)
        return;

    qDebug() << "C++ DEBUG: Starting Python AI:" << m_scriptPath;

    // 启动 python 进程
    QString program = "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3";
    // 注意：如果在 Windows 或某些环境可能需要改成 "python"

    QStringList args;
    args << "-u" << m_scriptPath; // -u 确保 Python stdout 不被缓冲

    m_proc.start(program, args);
    if (!m_proc.waitForStarted(3000)) {
        // [关键修复] 使用 errorOccured (一个r)
        emit errorOccured("Failed to start Python AI process: " + m_scriptPath);
        qDebug() << "C++ DEBUG: Start failed:" << m_proc.errorString();
        return;
    }
    m_started = true;
    qDebug() << "C++ DEBUG: Python AI started successfully.";
}

// 实现 AiClient::requestMove 接口
void PythonAiProcess::requestMove(const std::vector<std::vector<int>> &board,
                                int player,
                                std::optional<std::pair<int,int>> lastMove)
{
    Q_UNUSED(lastMove);
    if (!m_started)
        start();

    if (!m_started)  // 启动失败
        return;

    // C++ std::vector<std::vector<int>> 安全地转换成 Qt QVector<QVector<int>>
    QVector<QVector<int>> qtBoard;
    if (!board.empty()) {
        qtBoard.reserve(static_cast<int>(board.size()));
        for (const auto& row : board) {
            // [关键修复] 使用迭代器构造，彻底解决 fromStdVector 报错
            QVector<int> qtRow(row.begin(), row.end());
            qtBoard.append(qtRow);
        }
    }

    sendJsonRequest(qtBoard, player);
}

void PythonAiProcess::sendJsonRequest(const QVector<QVector<int>> &board, int who)
{
    int N = static_cast<int>(board.size());
    QJsonObject root;

    // board: List[List[int]]
    QJsonArray rows;
    for (int r = 0; r < N; ++r) {
        QJsonArray row;
        const auto& boardRow = board[r];
        int M = static_cast<int>(boardRow.size());

        for (int c = 0; c < M; ++c) {
            row.append(boardRow[c]);
        }
        rows.append(row);
    }
    root["board"] = rows;

    // who: 1=BLACK, 2=WHITE  → python 里是 "black"/"white"
    QString player = (who == 1) ? "black" : "white";
    root["player"] = player;

    QJsonDocument doc(root);
    QByteArray line = doc.toJson(QJsonDocument::Compact);
    line.append('\n');

    qDebug() << "C++ SEND:" << line; // 调试日志

    m_proc.write(line);
    m_proc.waitForBytesWritten();
}

void PythonAiProcess::onReadyRead()
{
    // 从 Python stdout 读取数据
    m_buffer.append(m_proc.readAllStandardOutput());

    // 简单按行切分，一行一个 json
    while (true) {
        int idx = m_buffer.indexOf('\n');
        if (idx < 0)
            break;
        QByteArray line = m_buffer.left(idx);
        m_buffer.remove(0, idx + 1);

        if (line.trimmed().isEmpty())
            continue;

        handleOneJsonLine(line);
    }
}

void PythonAiProcess::handleOneJsonLine(const QByteArray &line)
{
    qDebug() << "C++ RECV:" << line; // 调试日志

    QJsonParseError err;
    QJsonDocument doc = QJsonDocument::fromJson(line, &err);
    if (err.error != QJsonParseError::NoError || !doc.isObject()) {
        // [关键修复] 使用 errorOccured (一个r)
        emit errorOccured(QString("Bad JSON from AI: %1").arg(QString::fromUtf8(line)));
        return;
    }
    QJsonObject obj = doc.object();

    int row = obj.value("row").toInt(-1);
    int col = obj.value("col").toInt(-1);
    if (row < 0 || col < 0) {
        QString msg = obj.value("error").toString();
        if (msg.isEmpty())
            msg = "AI returned invalid move";
        // [关键修复] 使用 errorOccured (一个r)
        emit errorOccured(msg);
        return;
    }

    emit moveReady(row, col);
}

void PythonAiProcess::onProcessError(QProcess::ProcessError e)
{
    Q_UNUSED(e);
    qDebug() << "Python Process Error:" << m_proc.errorString();
    // [关键修复] 使用 errorOccured (一个r)
    emit errorOccured("Python AI process error: " + m_proc.errorString());
}

void PythonAiProcess::onFinished(int exitCode, QProcess::ExitStatus status)
{
    Q_UNUSED(exitCode);

    // --- 新增：读取并打印 Python 的标准错误输出 (Stderr) ---
    QByteArray errorOutput = m_proc.readAllStandardError();
    QString errorStr = QString::fromUtf8(errorOutput);

    if (!errorStr.isEmpty()) {
        qDebug() << "Python Stderr:" << errorStr; // 在 Qt Creator 控制台打印
    }
    // ---------------------------------------------------

    m_started = false;

    if (status == QProcess::CrashExit || !errorStr.isEmpty()) {
        qDebug() << "Python Process Crashed/Failed";
        // 把具体的错误信息弹窗告诉用户
        emit errorOccured("Python AI Error:\n" + errorStr);
    } else {
        qDebug() << "Python Process Finished Unexpectedly";
        emit errorOccured("Python AI process finished unexpectedly.");
    }
}