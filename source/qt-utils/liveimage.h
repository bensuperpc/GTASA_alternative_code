#ifndef LIVEIMAGE_H
#define LIVEIMAGE_H

#include <QImage>
#include <QPainter>
#include <QQuickPaintedItem>
#include <iostream>

class LiveImage : public QQuickPaintedItem
{
  Q_OBJECT
  Q_PROPERTY(QImage image MEMBER m_image WRITE setImage)
  Q_PROPERTY(bool enable_rescale READ enable_rescale WRITE set_enable_rescale NOTIFY enable_rescale_changed)
private:
  // Just storage for the image
  QImage m_image;
  bool m_enable_rescale = true;

public:
  explicit LiveImage(QQuickItem* parent = nullptr);
  void setImage(const QImage& image);
  void paint(QPainter* painter) override;

  Q_INVOKABLE bool enable_rescale() const { return m_enable_rescale; };

signals:
  void enable_rescale_changed(bool newValue);
public slots:
  void set_enable_rescale(bool value);
};

#endif  // SOURCE_LIVEIMAGE_H_
