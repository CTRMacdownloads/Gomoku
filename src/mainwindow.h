#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#include <QMainWindow>
class BoardWidget;
class GameModel;
class GameController;
class QAction;
class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
private slots:
    void onNewGame();
    void onUndo();
    void onStatusMsg(const QString &msg);
private:
    void createUi();
    void createMenu();
private:
    BoardWidget *m_board = nullptr;
    GameModel *m_model = nullptr;
    GameController *m_controller = nullptr;
    QAction *m_newGameAct = nullptr;
    QAction *m_undoAct = nullptr;
    QAction *m_quitAct = nullptr;
};
#endif // MAINWINDOW_H
