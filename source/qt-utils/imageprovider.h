#ifndef IMAGEPROVIDER_H
#define IMAGEPROVIDER_H

#include <QImage>
#include <QObject>
#include <QString>
#include <string>

class ImageProvider : public QObject
{
  Q_OBJECT
  Q_PROPERTY(QImage image MEMBER m_image READ image WRITE setImage NOTIFY imageChanged)

  QImage m_image;

public:
  explicit ImageProvider(QObject* parent = nullptr);
  void setImage(QImage const& image);
  void setImage(QString const& image);
  void setImage(std::string const& image);

  QImage image() const;

signals:
  void imageChanged();
};

#endif  // SOURCE_IMAGEPROVIDER_H_