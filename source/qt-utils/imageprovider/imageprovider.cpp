#include "imageprovider.h"

ImageProvider::ImageProvider() : QQuickImageProvider(QQuickImageProvider::Image, QQmlImageProviderBase::ForceAsynchronousImageLoading)
{

}

QPixmap ImageProvider::requestPixmap(const QString &id, QSize *size, const QSize &requestedSize)
{
    QPixmap pixmap =  QPixmap("/home/bensuperpc/Profile_400x400.jpg");
    return pixmap;
}

QImage ImageProvider::requestImage(const QString &id, QSize *size, const QSize &requestedSize)
{
    QImage image =  QImage("/home/bensuperpc/Profile_400x400.jpg");
    return image;
}
