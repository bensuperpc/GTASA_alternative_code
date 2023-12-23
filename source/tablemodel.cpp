#include "tablemodel.h"

TableModel::TableModel(QObject* parent) : QAbstractTableModel(parent) {
    QVector<QString> header = {"col0", "col1", "col2", "col3"};
    this->init(header);
    QVector<QString> data = {"data0", "data1", "data2", "data3"};
    this->addData(data);
}

void TableModel::clear() {
    this->table.clear();
    this->table.shrink_to_fit();
}

void TableModel::init(const QVector<QString>& vect) {
    this->addData(vect);
}

QVariant TableModel::data(const QModelIndex& index, int role) const {
    switch (role) {
        [[likely]] case TableDataRole : { return table.at(index.row()).at(index.column()); }
        case HeadingRole: {
            return index.row() == 0;
        }
            [[unlikely]] default : break;
    }
    return QVariant();
}

QHash<int, QByteArray> TableModel::roleNames() const {
    QHash<int, QByteArray> roles;
    roles[TableDataRole] = "tabledata";
    roles[HeadingRole] = "heading";
    return roles;
}

void TableModel::addData(const QVector<QString>& vect) {
    beginInsertRows(QModelIndex(), rowCount(), rowCount());
    // if (vect.size() != table
    table.append(vect);
    endInsertRows();
}

void TableModel::addData(const std::vector<QString>& vect) {
    beginInsertRows(QModelIndex(), rowCount(), rowCount());
#if QT_VERSION <= QT_VERSION_CHECK(5, 14, 0)
    table.append(QVector<QString>::fromStdVector(vect));
#else
    table.append(QVector<QString>(vect.begin(), vect.end()));
#endif
    endInsertRows();
}
