#include "gta_sa_ui.hpp"

GTA_SA_UI::GTA_SA_UI(std::unique_ptr<GTA_SA_MAIN> _gta_sa, TableModel& tableModel, QObject* parent)
    : QObject{parent}, selected_gta_sa(std::move(_gta_sa)), _tableModel(tableModel) {}

GTA_SA_UI::~GTA_SA_UI() {
    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}
void GTA_SA_UI::setMinRangeValue(uint64_t value) {
    std::cout << __FUNCTION_NAME__ << ": " << value << std::endl;

    selected_gta_sa->setMinRange(value);
    emit minRangeValueChanged(value);
}

void GTA_SA_UI::setMaxRangeValue(uint64_t value) {
    std::cout << __FUNCTION_NAME__ << ": " << value << std::endl;

    selected_gta_sa->setMaxRange(value);
    emit maxRangeValueChanged(value);
}

void GTA_SA_UI::setNbrThreadValue(uint32_t value) {
    std::cout << __FUNCTION_NAME__ << ": " << value << std::endl;

    selected_gta_sa->setThreadCount(value);
    emit nbrThreadValueChanged(value);
}

void GTA_SA_UI::set_cuda_block_size(uint64_t value) {
    std::cout << __FUNCTION_NAME__ << ": " << value << std::endl;

    selected_gta_sa->setCudaBlockSize(value);
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

    selected_gta_sa->swichMode(value);

    emit calc_mode_changed(value);
}

uint64_t GTA_SA_UI::calc_mode() const {
    return selected_gta_sa->getCurrentModeInt();
}

void GTA_SA_UI::runOpThread() {
    std::lock_guard<std::mutex> lock(_mtx);
    std::cout << __FUNCTION_NAME__ << std::endl;
    // Clear old data
    _tableModel.clear();

    // Launch operation
    selected_gta_sa->run();

    QVector<QString> header = {"Index", "Code", "JamCRC", "Associated code"};
    _tableModel.addData(header);

    /*
    // Store results in TableView Data
    for (const auto& result : selected_gta_sa->results) {
        QVector<QString> vect = {QString::number(result.index), QString::fromStdString(result.code),
                                 QString("0x") + QString::number(result.jamcrc, 16), QString::fromStdString(result.associated_code)};
        _tableModel.addData(vect);
    }
    */
    this->setButtonValue("  Launch Bruteforce   ");
}

void GTA_SA_UI::runOp() {
    this->setButtonValue("Bruteforce in progress");
    threads.emplace_back(&GTA_SA_UI::runOpThread, this);
}
