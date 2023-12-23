#include "gta_sa_ui.hpp"

GTA_SA_UI::GTA_SA_UI(std::unique_ptr<GTA_SA_Virtual> _gta_sa, TableModel& tableModel, QObject* parent) : QObject{parent}, selected_gta_sa(std::move(_gta_sa)), _tableModel(tableModel) {}

GTA_SA_UI::~GTA_SA_UI() {
    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}
void GTA_SA_UI::setMinRangeValue(uint64_t value) {
    std::cout << __FUNCTION_NAME__ << ": " << value << std::endl;

    if (selected_gta_sa->IsRunning) {
        std::cout << "Error: Can't change min range while running" << std::endl;
        return;
    }

    selected_gta_sa->min_range = value;
    emit minRangeValueChanged(value);
}

void GTA_SA_UI::setMaxRangeValue(uint64_t value) {
    std::cout << __FUNCTION_NAME__ << ": " << value << std::endl;

    if (selected_gta_sa->IsRunning) {
        std::cout << "Error: Can't change max range while running" << std::endl;
        return;
    }

    selected_gta_sa->max_range = value;
    emit maxRangeValueChanged(value);
}

void GTA_SA_UI::setNbrThreadValue(uint32_t value) {
    std::cout << __FUNCTION_NAME__ << ": " << value << std::endl;

    if (selected_gta_sa->IsRunning) {
        std::cout << "Error: Can't change number of thread while running" << std::endl;
        return;
    }

    selected_gta_sa->num_thread = value;
    emit nbrThreadValueChanged(value);
}

void GTA_SA_UI::set_cuda_block_size(uint64_t value) {
    std::cout << __FUNCTION_NAME__ << ": " << value << std::endl;

    if (selected_gta_sa->IsRunning) {
        std::cout << "Error: Can't change cuda block size while running" << std::endl;
        return;
    }

    selected_gta_sa->cuda_block_size = value;
    emit cuda_block_size_changed(value);
}

void GTA_SA_UI::setButtonValue(QString value) {
    std::cout << __FUNCTION_NAME__ << ": " << value.toStdString() << std::endl;
    if (value == _buttonValue)
        return;
    _buttonValue = value;
    emit buttonValueChanged(value);
}

void GTA_SA_UI::set_calc_mode(uint64_t value) {
    std::cout << __FUNCTION_NAME__ << ": " << value << std::endl;

    if (selected_gta_sa->IsRunning) {
        std::cout << "Error: Can't change compute type while running" << std::endl;
        return;
    }

    std::unique_ptr<GTA_SA_Virtual> tmp = nullptr;

    switch (value) {
        case 0: {
            std::cout << "Switching to STD_THREAD" << std::endl;
            tmp = std::move(std::make_unique<GTA_SA_STDTHREAD>());
            break;
        }
        case 1: {
            std::cout << "Switching to OPEN_MP" << std::endl;
            tmp = std::move(std::make_unique<GTA_SA_OPENMP>());
            break;
        }
        case 2: {
            std::cout << "Switching to CUDA" << std::endl;
#if defined(BUILD_WITH_CUDA)
            tmp = std::move(std::make_unique<GTA_SA_CUDA>());
#else
            std::cout << "CUDA not supported" << std::endl;
#endif
            break;
        }
        case 3: {
            std::cout << "Switching to OPENCL" << std::endl;
            // tmp = std::move(std::make_unique<GTA_SA_OPENCL>());
            break;
        }
        default: {
            std::cout << "Switching to STD_THREAD" << std::endl;
            tmp = std::move(std::make_unique<GTA_SA_STDTHREAD>());
            break;
        }
    }

    if (tmp == nullptr) {
        std::cout << "Error: Invalid or not implemented compute type:" << value << std::endl;
        return;
    }

    if (typeid(*selected_gta_sa).hash_code() == typeid(*tmp).hash_code()) {
        std::cout << "Error: Already using this compute type:" << value << std::endl;
        return;
    }

    tmp->min_range = selected_gta_sa->min_range;
    tmp->max_range = selected_gta_sa->max_range;
    tmp->num_thread = selected_gta_sa->num_thread;
    tmp->cuda_block_size = selected_gta_sa->cuda_block_size;

    this->selected_gta_sa = std::move(tmp);

    emit calc_mode_changed(value);
}

uint64_t GTA_SA_UI::calc_mode() const {
    if (typeid(*selected_gta_sa).hash_code() == typeid(GTA_SA_STDTHREAD).hash_code()) {
        return 0;
    } else if (typeid(*selected_gta_sa).hash_code() == typeid(GTA_SA_OPENMP).hash_code()) {
        return 1;
#if defined(BUILD_WITH_CUDA)
    } else if (typeid(*selected_gta_sa).hash_code() == typeid(GTA_SA_CUDA).hash_code()) {
        return 2;
#endif
    } else {
        return 0;
    }
}

void GTA_SA_UI::runOpThread() {
    std::lock_guard<std::mutex> lock(_mtx);
    std::cout << __FUNCTION_NAME__ << std::endl;
    // Clear old data
    selected_gta_sa->clear();
    _tableModel.clear();

    // Launch operation
    selected_gta_sa->run();

    QVector<QString> header = {"Index", "Code", "JamCRC", "Associated code"};
    _tableModel.addData(header);

    // Store results in TableView Data
    for (const auto& result : selected_gta_sa->results) {
        QVector<QString> vect = {QString::number(result.index), QString::fromStdString(result.code),
                                 QString("0x") + QString::number(result.jamcrc, 16), QString::fromStdString(result.associated_code)};
        _tableModel.addData(vect);
    }

    this->setButtonValue("  Launch Bruteforce   ");
}

void GTA_SA_UI::runOp() {
    this->setButtonValue("Bruteforce in progress");
    threads.emplace_back(&GTA_SA_UI::runOpThread, this);
}
