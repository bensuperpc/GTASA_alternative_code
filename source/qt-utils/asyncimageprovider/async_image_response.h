#ifndef ASYNCIMAGEPROVIDER_ASYNC_IMAGE_RESPONSE_H
#define ASYNCIMAGEPROVIDER_ASYNC_IMAGE_RESPONSE_H

#include <qqmlengine.h>
#include <qquickimageprovider.h>
#include <QDebug>
#include <QImage>
#include <QThreadPool>

#include "async_image_response_runnable.h"

class AsyncImageResponse : public QQuickImageResponse
{
public:
    AsyncImageResponse(const QString &id, const QSize &requestedSize, QThreadPool *pool)
    {
        auto runnable = new AsyncImageResponseRunnable(id, requestedSize);
        connect(runnable, &AsyncImageResponseRunnable::done, this, &AsyncImageResponse::handleDone);
        pool->start(runnable);
    }

    void handleDone(QImage image) {
        m_image = image;
        emit finished();
    }

    QQuickTextureFactory *textureFactory() const override
    {
        return QQuickTextureFactory::textureFactoryForImage(m_image);
    }

    QImage m_image;
};

#endif // ASYNCIMAGEPROVIDER_ASYNC_IMAGE_RESPONSE_RUNNABLE_H