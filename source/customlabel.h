#ifndef MYLABEL_H
#define MYLABEL_H

#include <QFont>
#include <QObject>
#include <QVariant>

class MyLabel : public QObject {
    Q_OBJECT
   public:
    explicit MyLabel(QObject* parent = nullptr);

    Q_INVOKABLE void setMyObject(QObject* obj);
   signals:

   private:
    QObject* myObject;
};

#endif  // MYLABEL_H
