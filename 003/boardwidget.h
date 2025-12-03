#ifndef BOARDWIDGET_H
#define BOARDWIDGET_H
#include <QWidget>
#include <optional>
class GameModel;
class BoardWidget : public QWidget {
    Q_OBJECT
public:
    explicit BoardWidget(QWidget *parent = nullptr);
    void setModel(GameModel *model);
    QSize sizeHint() const override { return {640, 640}; }
signals:
    void cellClicked(int row, int col);
protected:
    void paintEvent(QPaintEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
private:
    QRect gridRect() const;
    std::pair<int,int> posToCell(const QPoint &p) const;
private:
    GameModel *m_model = nullptr;
    int m_margin = 30;
};
#endif // BOARDWIDGET_H
