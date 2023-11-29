// Base on: https://code.qt.io/cgit/qt/qtcharts.git/tree/examples/charts/qmloscilloscope?h=6.4

#ifndef LINESERIES_H
#define LINESERIES_H

#include <QDebug>
#include <QLineSeries>
#include <QObject>
#include <cmath>
#include <random>
#include <vector>

#include <QtCharts/QXYSeries>

class LineSeries : public QObject {
    Q_OBJECT
   public:
    explicit LineSeries(QObject* parent = nullptr);

   signals:

   public slots:
    // Q_INVOKABLE
    void generateData(const int rowCount, const int colCount);
    // Q_INVOKABLE
    void update(QAbstractSeries* series);

   private:
    QList<QList<QPointF>> m_data;
    int m_index;
};

#endif  // LINESERIES_H
