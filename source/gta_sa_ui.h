#ifndef _GTA_SA_UI_H_
#define _GTA_SA_UI_H_

#include <QEventLoop>
#include <QObject>
#include <QString>
#include <iostream>
#include <thread>
#include <vector>

#include "GTA_SA_cheat_finder.hpp"
#include "qt-utils/TableModel.h"
#include "utils/utils.h"

class GTA_SA_UI : public QObject
{
  Q_OBJECT
  Q_PROPERTY(uint64_t minRangeValue READ minRangeValue WRITE setMinRangeValue NOTIFY minRangeValueChanged)
  Q_PROPERTY(uint64_t maxRangeValue READ maxRangeValue WRITE setMaxRangeValue NOTIFY maxRangeValueChanged)

  Q_PROPERTY(uint32_t nbrThreadValue READ nbrThreadValue WRITE setNbrThreadValue NOTIFY nbrThreadValueChanged)
  Q_PROPERTY(uint64_t cuda_block_size READ cuda_block_size WRITE set_cuda_block_size NOTIFY cuda_block_size_changed)

  Q_PROPERTY(QString buttonValue READ buttonValue WRITE setButtonValue NOTIFY buttonValueChanged)

  Q_PROPERTY(uint64_t calc_mode READ calc_mode WRITE set_calc_mode NOTIFY calc_mode_changed)

  Q_PROPERTY(bool builtWithOpenMP READ builtWithOpenMP CONSTANT)
  Q_PROPERTY(bool builtWithCUDA READ builtWithCUDA CONSTANT)

public:
  explicit GTA_SA_UI(QObject* parent = nullptr);
  GTA_SA gta_sa;
  TableModel tableModel;

  uint64_t minRangeValue() const { return _minRangeValue; };
  uint64_t maxRangeValue() const { return _maxRangeValue; };

  uint64_t nbrThreadValue() const { return _nbrThreadValue; };

  uint64_t cuda_block_size() const { return _cuda_block_size; };

  uint64_t calc_mode() const { return _calc_mode; };

  QString buttonValue() const { return _buttonValue; };

  Q_INVOKABLE
  void runOp();
  void runOpThread();

  Q_INVOKABLE
  uint64_t max_thread_support() { return gta_sa.max_thread_support(); }

  bool builtWithOpenMP() const { return GTA_SA::builtWithOpenMP; };

  bool builtWithCUDA() const { return GTA_SA::builtWithCUDA; };

public slots:
  void setMinRangeValue(uint64_t value);
  void setMaxRangeValue(uint64_t value);

  void setNbrThreadValue(uint32_t value);
  void set_cuda_block_size(uint64_t value);

  void set_calc_mode(uint64_t value);

  void setButtonValue(QString value);

signals:
  void minRangeValueChanged(uint64_t value);
  void maxRangeValueChanged(uint64_t value);

  void nbrThreadValueChanged(uint32_t value);
  void cuda_block_size_changed(uint64_t value);
  void calc_mode_changed(uint64_t value);

  void buttonValueChanged(QString value);

private:
  uint64_t& _minRangeValue = gta_sa.min_range;
  uint64_t& _maxRangeValue = gta_sa.max_range;
  QString _buttonValue = "Launch Bruteforce";
  uint32_t& _nbrThreadValue = gta_sa.num_thread;
  uint64_t& _cuda_block_size = gta_sa.cuda_block_size;
  uint64_t& _calc_mode = gta_sa.calc_mode;
};

#endif  // GTA_SA_UI_H
