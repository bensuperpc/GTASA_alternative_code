#ifndef ASYNCIMAGEPROVIDER_ASYNC_IMAGE_RESPONSE_RUNNABLE_H
#define ASYNCIMAGEPROVIDER_ASYNC_IMAGE_RESPONSE_RUNNABLE_H

#include <qqmlengine.h>
#include <qquickimageprovider.h>
#include <QDebug>
#include <QImage>
#include <QThreadPool>

class AsyncImageResponseRunnable : public QObject, public QRunnable
{
    Q_OBJECT

signals:
    void done(QImage image);

public:
    AsyncImageResponseRunnable(const QString &id, const QSize &requestedSize)
        : m_id(id), m_requestedSize(requestedSize) {}

    void run() override
    {
        if (m_id == QLatin1String("slow")) {
            qDebug() << "Slow, red, sleeping for 5 seconds";
            QThread::sleep(5);
        } else {
            qDebug() << "Fast, blue, sleeping for 1 second";
            QThread::sleep(1);
        }
        QImage image =  QImage("/home/bedem/Profile_400x400.jpg");

        if (m_requestedSize.isValid())
            image = image.scaled(m_requestedSize);

        emit done(image);
    }

private:
    QString m_id;
    QSize m_requestedSize;
};

#endif // ASYNCIMAGEPROVIDER_H