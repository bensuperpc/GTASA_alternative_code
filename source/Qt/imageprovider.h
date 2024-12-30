#ifndef IMAGEPROVIDER_H
#define IMAGEPROVIDER_H

#include <QQuickImageProvider>
#include <iostream>

class ImageProvider final : public QQuickImageProvider {
   public:
    explicit ImageProvider();

    QPixmap requestPixmap(const QString& id, QSize* size, const QSize& requestedSize) override;
    QImage requestImage(const QString& id, QSize* size, const QSize& requestedSize) override;
};

#endif  // IMAGEPROVIDER_H
