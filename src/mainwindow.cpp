#include "mainwindow.h"
#include "boardwidget.h"
#include "gamemodel.h"
#include "gamecontroller.h"
#include "python_ai_process.h"
#include <QMenuBar>
#include <QMenu>
#include <QStatusBar>
#include <QAction>
#include <QInputDialog>
MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    createUi();
    createMenu();
    statusBar()->showMessage("Gomoku (Qt6) ready");
}
MainWindow::~MainWindow() { delete m_controller; delete m_model; }
void MainWindow::createUi() {
    m_board = new BoardWidget(this);
    setCentralWidget(m_board);
    m_model = new GameModel(15);
    m_board->setModel(m_model);
    m_controller = new GameController(m_model, m_board, this);
    connect(m_controller, &GameController::messageChanged,
            this, &MainWindow::onStatusMsg);
    resize(800, 800);
}
void MainWindow::createMenu() {
    auto *gameMenu = menuBar()->addMenu(tr("&Game"));
    m_newGameAct = new QAction(tr("&New Game"), this);
    m_newGameAct->setShortcut(QKeySequence::New);
    connect(m_newGameAct, &QAction::triggered, this, &MainWindow::onNewGame);
    m_undoAct = new QAction(tr("&Undo"), this);
    m_undoAct->setShortcut(QKeySequence::Undo);
    connect(m_undoAct, &QAction::triggered, this, &MainWindow::onUndo);
    m_quitAct = new QAction(tr("&Quit"), this);
    m_quitAct->setShortcut(QKeySequence::Quit);
    connect(m_quitAct, &QAction::triggered, this, &QWidget::close);
    gameMenu->addAction(m_newGameAct);
    gameMenu->addAction(m_undoAct);
    gameMenu->addSeparator();
    gameMenu->addAction(m_quitAct);
}
void MainWindow::onNewGame() {
    bool ok = false;
    int size = QInputDialog::getInt(this, tr("Board Size"),
                                    tr("Size (>= 5):"), 15, 5, 25, 1, &ok);
    if (ok) { m_controller->newGame(size); }
}
void MainWindow::onUndo() { m_controller->undo(); }
void MainWindow::onStatusMsg(const QString &msg) { statusBar()->showMessage(msg, 3000); }
