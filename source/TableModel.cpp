#include "TableModel.h"

void TableModel::clear()
{
  this->table.clear();
  this->table.shrink_to_fit();
  this->init();
}

void TableModel::init()
{
  QVector<QString> vect = {"Iter. N°", "Code", "JAMCRC value", "GTA Code"};
  this->addPerson(vect);
}

QVariant TableModel::data(const QModelIndex& index, int role) const
{
  switch (role) {
    case TableDataRole: {
      return table.at(index.row()).at(index.column());
    }
    case HeadingRole: {
      return index.row() == 0;
    }
    default:
      break;
  }
  return QVariant();
}

QHash<int, QByteArray> TableModel::roleNames() const
{
  QHash<int, QByteArray> roles;
  roles[TableDataRole] = "tabledata";
  roles[HeadingRole] = "heading";
  return roles;
}

void TableModel::addPerson(const QVector<QString>& vect)
{
  beginInsertRows(QModelIndex(), rowCount(), rowCount());
  table.append(vect);
  endInsertRows();
}

void TableModel::addPerson(const std::vector<QString>& vect)
{
  beginInsertRows(QModelIndex(), rowCount(), rowCount());
#if QT_VERSION <= QT_VERSION_CHECK(5, 14, 0)
  table.append(QVector<QString>::fromStdVector(vect));
#else
  table.append(QVector<QString>(vect.begin(), vect.end()));
#endif
  endInsertRows();
}