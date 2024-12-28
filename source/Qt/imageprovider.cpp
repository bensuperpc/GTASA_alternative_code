#include "imageprovider.h"

ImageProvider::ImageProvider() : QQuickImageProvider(QQuickImageProvider::Image, QQmlImageProviderBase::ForceAsynchronousImageLoading) {}

QPixmap ImageProvider::requestPixmap(const QString& id, QSize* size, const QSize& requestedSize) {
    const QString file = "image/" + id;
    QPixmap pixmap = QPixmap(file);
    return pixmap;
}

QImage ImageProvider::requestImage(const QString& id, QSize* size, const QSize& requestedSize) {
    const QString file = "image/" + id;
    QImage image = QImage(file);
    return image;
}
