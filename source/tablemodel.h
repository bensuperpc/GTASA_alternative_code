#ifndef TABLEMODEL_H
#define TABLEMODEL_H

#include <QAbstractTableModel>
#include <QObject>

class TableModel : public QAbstractTableModel {
    Q_OBJECT
    enum TableRoles { TableDataRole = Qt::UserRole + 1, HeadingRole };

   public:
    explicit TableModel(QObject* parent = nullptr);

    void clear();

    void init(const QVector<QString>& vect);

    Q_INVOKABLE
    int rowCount(const QModelIndex& = QModelIndex()) const override { return static_cast<int>(table.size()); }

    Q_INVOKABLE
    int columnCount(const QModelIndex& = QModelIndex()) const override { return static_cast<int>(table.at(0).size()); }

    QVariant data(const QModelIndex& index, int role) const override;

    QHash<int, QByteArray> roleNames() const override;

    Q_INVOKABLE void addData(const QVector<QString>& vect);

    Q_INVOKABLE void addData(const std::vector<QString>& vect);

   private:
    QVector<QVector<QString>> table;
};

#endif
