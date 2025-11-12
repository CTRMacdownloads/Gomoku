#include "python_ai_process.h"
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QCoreApplication>

PythonAiProcess::PythonAiProcess(const QString& pythonExe,
                                 const QString& scriptPath,
                                 const int msecs,
                                 QObject* parent)
    : AiClient(parent)
{
    QString absolutePath = QFileInfo(scriptPath).isRelative() ? QCoreApplication::applicationDirPath() + "/" + scriptPath : scriptPath;
    m_proc.setProgram(pythonExe);
    m_proc.setArguments({QStringLiteral("-u"), absolutePath});
    m_proc.setProcessChannelMode(QProcess::SeparateChannels);
    connect(&m_proc, &QProcess::readyReadStandardOutput, this, &PythonAiProcess::onReadyRead);
    connect(&m_proc, &QProcess::readyReadStandardError, this, [this]{
        const QByteArray err = m_proc.readAllStandardError();
        emit errorOccured(QString::fromLocal8Bit(err));
    });
    connect(&m_proc, &QProcess::errorOccurred, this, &PythonAiProcess::onError);
    m_proc.start();
    m_proc.waitForStarted(msecs);
}

void PythonAiProcess::requestMove(const std::vector<std::vector<int>>& board,
                                  int player, std::optional<std::pair<int,int>> lastMove)
{
    if (m_proc.state() != QProcess::Running) {
        emit errorOccured(QStringLiteral("AI process not running"));
        return;
    }
    QJsonObject obj;
    obj["size"] = int(board.size());
    QJsonArray rows;
    for (const auto& r : board) {
        QJsonArray ja;
        for (int v : r) ja.append(v);
        rows.append(ja);
    }
    obj["board"] = rows;
    obj["player"] = (player == 1 ? "black" : "white");
    if (lastMove.has_value()) {
        QJsonArray lm;
        lm.append(lastMove->first);
        lm.append(lastMove->second);
        obj["last"] = lm;
    }
    auto line = QJsonDocument(obj).toJson(QJsonDocument::Compact) + "\n";
    m_proc.write(line);
    m_proc.waitForBytesWritten();   // block the main thread until json data has totally been written into std buffer
}

void PythonAiProcess::onReadyRead() {
    m_buffer += m_proc.readAllStandardOutput();
    long long idx;
    while ((idx = m_buffer.indexOf('\n')) != -1) {
        QByteArray line = m_buffer.left(idx);
        m_buffer.remove(0, idx + 1);
        QJsonParseError err{};
        QJsonDocument doc = QJsonDocument::fromJson(line, &err);
        if (err.error != QJsonParseError::NoError || !doc.isObject()) {
            emit errorOccured(QStringLiteral("AI JSON parse error"));
            continue;
        }
        QJsonObject obj = doc.object();
        int r = obj.value("row").toInt(-1);
        int c = obj.value("col").toInt(-1);
        if (r >= 0 && c >= 0) {
            emit moveReady(r, c);
        }
        else {
            emit errorOccured(QStringLiteral("AI returned invalid move"));
        }
    }
}

void PythonAiProcess::onError(QProcess::ProcessError) {
    emit errorOccured(QStringLiteral("AI process error: %1").arg(m_proc.errorString()));
}
