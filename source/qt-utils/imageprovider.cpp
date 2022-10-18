#include "imageprovider.h"

ImageProvider::ImageProvider(QObject* parent)
    : QObject(parent)
{
}

void ImageProvider::setImage(QImage const& image)
{
  m_image = image;
  emit imageChanged();
}

void ImageProvider::setImage(std::string const& image)
{
  setImage(QString::fromStdString(image));
}

void ImageProvider::setImage(QString const& image)
{
  QImage _image;
  _image.load(image);
  _image = _image.convertToFormat(QImage::Format_ARGB32);
  m_image = _image;

  emit imageChanged();
}

QImage ImageProvider::image() const
{
  return m_image;
}
