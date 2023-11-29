// Base on: https://code.qt.io/cgit/qt/qtcharts.git/tree/examples/charts/qmloscilloscope?h=6.4

#include "lineseries.h"

LineSeries::LineSeries(QObject* parent) : QObject{parent}, m_index(-1) {
    generateData(16, 1024);
}

void LineSeries::update(QAbstractSeries* series) {
    if (!series) {
        return;
    }

    m_index++;

    if (m_index > m_data.count() - 1) {
        m_index = 0;
    }

    QXYSeries* xySeries = static_cast<QXYSeries*>(series);

    QList<QPointF> points = m_data.at(m_index);

    xySeries->replace(points);
}

void LineSeries::generateData(const int rowCount, const int colCount) {
    m_data.clear();

    std::random_device rd;
    std::seed_seq seed{rd(), rd()};
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    for (int i = 0; i < rowCount; i++) {
        QList<QPointF> points;
        points.reserve(colCount);
        for (int i = 0; i < colCount; i++) {
            const qreal x = i;
            const qreal y = qSin(M_PI / 50 * i) + 0.5 + uni(rng);
            points.append(QPointF(x, y));
        }
        m_data.append(points);
    }
}
