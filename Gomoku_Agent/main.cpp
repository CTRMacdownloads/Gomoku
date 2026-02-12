#include <QApplication>
#include "mainwindow.h"
int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    MainWindow w;
    w.setWindowTitle("Gomoku - Qt6 (AI)");
    w.show();
    return app.exec();
}
