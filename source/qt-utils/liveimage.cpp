#include "liveimage.h"

LiveImage::LiveImage(QQuickItem* parent)
    : QQuickPaintedItem(parent)
    , m_image {}
{
}

void LiveImage::paint(QPainter* painter)
{
  qDebug() << Q_FUNC_INFO;
  if (m_image.isNull()) {
    return;
  }
  if (this->enable_rescale() == true) {
    const auto bounding_rect = boundingRect();  // QRectF
    // const auto scaled = m_image.scaledToHeight(bounding_rect.height());  // QImage
    const auto scaled =
        m_image.scaled(static_cast<int>(bounding_rect.width()), static_cast<int>(bounding_rect.height()));  // QImage
    auto center = bounding_rect.center() - scaled.rect().center();  // QPointF

    if (center.x() < 0)
      center.setX(0);
    if (center.y() < 0)
      center.setY(0);
    painter->drawImage(center, scaled);
  } else {
    painter->drawImage(0, 0, m_image);
  }

  /*
  // Update the image
  const auto size = this->size();
  const auto image_size = this->m_image.size();

  if ((size.width() < image_size.width() || size.height() < image_size.height())
      && (size.width() != 0 || size.height() != 0))
  {
    auto facteur = 1.0;

    auto hauteur = static_cast<float>(image_size.height())
        / static_cast<float>(size.height());
    auto largeur = static_cast<float>(image_size.width())
        / static_cast<float>(size.width());
    facteur = (largeur < hauteur) ? hauteur : largeur;

    painter->drawImage(
        0,
        0,
        this->m_image.scaled((static_cast<int>(image_size.width()) / facteur),
                       (static_cast<int>(image_size.height()) / facteur)));
  } else {
    painter->drawImage(0, 0, m_image);
  }
  */
}

void LiveImage::setImage(const QImage& image)
{
  qDebug() << Q_FUNC_INFO;
  if (image == m_image)
    return;
  m_image = image;

  std::cout << "Image updated" << std::endl;
  // Redraw the image
  update();
}

void LiveImage::set_enable_rescale(bool newValue)
{
  qDebug() << Q_FUNC_INFO;
  if (m_enable_rescale != newValue) {
    m_enable_rescale = newValue;
    emit enable_rescale_changed(newValue);
  }
  update();
}
