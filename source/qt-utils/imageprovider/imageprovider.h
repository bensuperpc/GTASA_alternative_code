#ifndef IMAGEPROVIDER_H
#define IMAGEPROVIDER_H

#include <iostream>

#include <QQuickImageProvider>

class ImageProvider : public QQuickImageProvider
{
public:
    ImageProvider();

    QPixmap requestPixmap(const QString &id, QSize *size, const QSize &requestedSize) override;
    QImage requestImage(const QString &id, QSize *size, const QSize &requestedSize) override;
};

#endif // IMAGEPROVIDER_H
