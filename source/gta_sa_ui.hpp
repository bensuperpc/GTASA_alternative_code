#ifndef _GTA_SA_UI_H_
#define _GTA_SA_UI_H_

#include <QEventLoop>
#include <QObject>
#include <QString>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "GTA_SA_cheat_finder_virtual.hpp"

#ifdef BUILD_WITH_CUDA
#include "GTA_SA_cheat_finder_cuda.hpp"
#endif  // BUILD_WITH_CUDA

#include "GTA_SA_cheat_finder_openmp.hpp"
#include "GTA_SA_cheat_finder_stdthread.hpp"

#include "tablemodel.h"
#include "utils/utils.h"

class GTA_SA_UI final : public QObject {
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
    GTA_SA_UI(QObject* parent = nullptr) = delete;
    explicit GTA_SA_UI(std::unique_ptr<GTA_SA_Virtual> _gta_sa, TableModel& tableModel, QObject* parent = nullptr);

    ~GTA_SA_UI();

    std::unique_ptr<GTA_SA_Virtual> selected_gta_sa;
    TableModel& _tableModel;

    uint64_t minRangeValue() const { return selected_gta_sa->min_range; }
    uint64_t maxRangeValue() const { return selected_gta_sa->max_range; }

    uint32_t nbrThreadValue() const { return selected_gta_sa->num_thread; }

    uint64_t cuda_block_size() const { return selected_gta_sa->cuda_block_size; }

    uint64_t calc_mode() const;

    QString buttonValue() const { return _buttonValue; }

    void set_gta_sa(std::unique_ptr<GTA_SA_Virtual>& _gta_sa) { this->selected_gta_sa = std::move(_gta_sa); }

    Q_INVOKABLE
    void runOp();
    void runOpThread();

    Q_INVOKABLE
    uint64_t max_thread_support() { return GTA_SA_Virtual::max_thread_support(); }

    bool builtWithOpenMP() const { return GTA_SA_Virtual::builtWithOpenMP; }

    bool builtWithCUDA() const { return GTA_SA_Virtual::builtWithCUDA; }

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
    QString _buttonValue = "Launch Bruteforce";
    std::vector<std::thread> threads;
    std::mutex _mtx;
};

#endif  // GTA_SA_UI_H
