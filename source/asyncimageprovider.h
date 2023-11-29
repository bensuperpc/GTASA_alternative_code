#ifndef ASYNCIMAGEPROVIDER_H
#define ASYNCIMAGEPROVIDER_H

#include <QDebug>
#include <QFile>
#include <QImage>
#include <QThreadPool>

#include <qqmlengine.h>
#include <qquickimageprovider.h>

class AsyncImageResponseRunnable : public QObject, public QRunnable {
    Q_OBJECT

   signals:
    void done(QImage image);

   public:
    AsyncImageResponseRunnable(const QString& id, const QSize& requestedSize) : m_id(id), m_requestedSize(requestedSize) {}

    void run() override {
        qDebug() << "Load image ID: " << m_id;
        // QThread::sleep(5);

        const QString file = "image/" + m_id;
        QImage image;
        if (QFile::exists(file)) {
            image.load(file);
        } else {
            // image = QImage(50, 50, QImage::Format_RGB32);
            // image.fill(Qt::white);
            qWarning() << "Image: " << file << " not found!";
        }

        if (m_requestedSize.isValid()) {
            image = image.scaled(m_requestedSize);
        }

        emit done(image);
    }

   private:
    QString m_id;
    QSize m_requestedSize;
};

class AsyncImageResponse : public QQuickImageResponse {
   public:
    AsyncImageResponse(const QString& id, const QSize& requestedSize, QThreadPool* pool) {
        auto runnable = new AsyncImageResponseRunnable(id, requestedSize);
        connect(runnable, &AsyncImageResponseRunnable::done, this, &AsyncImageResponse::handleDone);
        pool->start(runnable);
    }

    void handleDone(QImage image) {
        m_image = image;
        emit finished();
    }

    QQuickTextureFactory* textureFactory() const override { return QQuickTextureFactory::textureFactoryForImage(m_image); }

    QImage m_image;
};

class AsyncImageProvider : public QQuickAsyncImageProvider {
   public:
    QQuickImageResponse* requestImageResponse(const QString& id, const QSize& requestedSize) override {
        AsyncImageResponse* response = new AsyncImageResponse(id, requestedSize, &pool);
        return response;
    }

   private:
    QThreadPool pool;
};

#endif  // ASYNCIMAGEPROVIDER_H
