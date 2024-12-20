#include "gta_sa_ui.hpp"

GTA_SA_UI::GTA_SA_UI(std::unique_ptr<GTA_SA_ENGINE> _gta_sa, QObject* parent)
    : QObject{parent}, _mainCalculator(std::move(_gta_sa)) {}

GTA_SA_UI::~GTA_SA_UI() {
    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}
void GTA_SA_UI::setMinRangeValue(uint64_t value) {
    std::cout << __FUNCTION_NAME__ << ": " << value << std::endl;

    _mainCalculator->setMinRange(value);
    emit minRangeValueChanged(value);
}

void GTA_SA_UI::setMaxRangeValue(uint64_t value) {
    std::cout << __FUNCTION_NAME__ << ": " << value << std::endl;

    _mainCalculator->setMaxRange(value);
    emit maxRangeValueChanged(value);
}

void GTA_SA_UI::setNbrThreadValue(uint32_t value) {
    std::cout << __FUNCTION_NAME__ << ": " << value << std::endl;

    _mainCalculator->setThreadCount(value);
    emit nbrThreadValueChanged(value);
}

void GTA_SA_UI::set_cuda_block_size(uint64_t value) {
    std::cout << __FUNCTION_NAME__ << ": " << value << std::endl;

    _mainCalculator->setCudaBlockSize(value);
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

    _mainCalculator->swichMode(value);

    emit calc_mode_changed(value);
}

uint64_t GTA_SA_UI::calc_mode() const {
    return _mainCalculator->getCurrentModeInt();
}

void GTA_SA_UI::runOpThread() {
    std::lock_guard<std::mutex> lock(_mtx);
    std::cout << __FUNCTION_NAME__ << std::endl;
    this->setButtonValue("  Launch Bruteforce   ");
    _mainCalculator->run();
}

void GTA_SA_UI::runOp() {
    this->setButtonValue("Bruteforce in progress");
    threads.emplace_back(&GTA_SA_UI::runOpThread, this);
}
