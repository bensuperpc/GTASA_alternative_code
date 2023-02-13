#ifndef ASYNCIMAGEPROVIDER_H
#define ASYNCIMAGEPROVIDER_H

#include <qqmlengine.h>
#include <qquickimageprovider.h>
#include <QDebug>
#include <QImage>
#include <QThreadPool>

#include "async_image_response_runnable.h"
#include "async_image_response.h"

class AsyncImageProvider : public QQuickAsyncImageProvider
{
public:
    QQuickImageResponse *requestImageResponse(const QString &id, const QSize &requestedSize) override
    {
        AsyncImageResponse *response = new AsyncImageResponse(id, requestedSize, &pool);
        return response;
    }

private:
    QThreadPool pool;
};


#endif // ASYNCIMAGEPROVIDER_H
