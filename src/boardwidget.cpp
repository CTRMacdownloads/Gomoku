#include "boardwidget.h"
#include "gamemodel.h"
#include <QPainter>
#include <QMouseEvent>
#include <QPen>
#include <QBrush>
BoardWidget::BoardWidget(QWidget *parent) : QWidget(parent) {
    setMinimumSize(480, 480);
    setMouseTracking(true);
    setAttribute(Qt::WA_OpaquePaintEvent);
}

void BoardWidget::setModel(GameModel *model) {
    m_model = model;
    update();
}

QRect BoardWidget::gridRect() const {
    int side = qMin(width(), height()) - 2 * m_margin;
    int x = (width() - side) / 2;
    int y = (height() - side) / 2;
    return {x, y, side, side};
}

std::pair<int,int> BoardWidget::posToCell(const QPoint &p) const {
    if (!m_model) {
        return {-1, -1};
    }
    auto rect = gridRect();
    int N = m_model->size();
    if (!rect.contains(p)) {
        return {-1, -1};
    }
    const double step = rect.width() * 1.0 / (N - 1);
    double rx = (p.x() - rect.left()) / step;
    double ry = (p.y() - rect.top()) / step;
    int c = qRound(rx);
    int r = qRound(ry);
    if (r < 0 || r >= N || c < 0 || c >= N) {
        return {-1, -1};
    }
    QPointF cross(rect.left() + c * step, rect.top() + r * step);
    if (QLineF(p, cross).length() > step * 0.4) {
        return {-1, -1};
    }
    return {r, c};
}

void BoardWidget::paintEvent(QPaintEvent *) {
    QPainter g(this);
    g.setRenderHint(QPainter::Antialiasing, true);
    g.fillRect(rect(), QColor(240, 240, 240));
    if (!m_model) {
        return;
    }
    auto r = gridRect();
    g.fillRect(r, QColor(235, 200, 150));
    QPen pen(Qt::black);
    pen.setWidth(1);
    g.setPen(pen);
    int N = m_model->size();
    const double step = r.width() * 1.0 / (N - 1);
    for (int i = 0; i < N; ++i) {
        int x1 = r.left();
        int x2 = r.right();
        int y = r.top() + i * step;
        g.drawLine(x1, y, x2, y);
        int y1 = r.top();
        int y2 = r.bottom();
        int x = r.left() + i * step;
        g.drawLine(x, y1, x, y2);
    }
    auto drawStar = [&](int rr, int cc){
        QPointF p(r.left() + cc*step, r.top() + rr*step);
        g.setBrush(Qt::black);
        g.drawEllipse(p, 3, 3);
    };
    if (N >= 15) {
        int s = N/2;
        drawStar(s, s);
        int k = 3;
        drawStar(k, k); drawStar(k, N-1-k);
        drawStar(N-1-k, k); drawStar(N-1-k, N-1-k);
    }
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            auto st = m_model->stoneAt(i, j);
            if (st == GameModel::Empty) {
                continue;
            }
            QPointF center(r.left() + j*step, r.top() + i*step);
            double rad = step * 0.42;
            g.setPen(Qt::NoPen);
            g.setBrush(st == GameModel::Black ? Qt::black : Qt::white);
            g.drawEllipse(center, rad, rad);
            QPen edge(Qt::black);
            edge.setWidthF(1.0);
            g.setPen(edge);
            g.setBrush(Qt::NoBrush);
            g.drawEllipse(center, rad, rad);
        }
    }
    if (m_model->lastMove().has_value()) {
        auto [lr, lc] = *m_model->lastMove();
        QPointF center(r.left() + lc*step, r.top() + lr*step);
        double rad = step * 0.18;
        QPen hi(Qt::red);
        hi.setWidth(2);
        g.setPen(hi);
        g.setBrush(Qt::NoBrush);
        g.drawEllipse(center, rad, rad);
    }
}

void BoardWidget::mousePressEvent(QMouseEvent *event) {
    if (!m_model) return;
    auto [r, c] = posToCell(event->pos());
    if (r >= 0) emit cellClicked(r, c);
}
