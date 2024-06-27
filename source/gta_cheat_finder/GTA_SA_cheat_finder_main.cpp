#include "GTA_SA_cheat_finder_main.hpp"

GTA_SA_MAIN::GTA_SA_MAIN() {
    _threadCount = GTA_SA_Virtual::maxThreadSupport();
    _cudaBlockSize = 64;
}

GTA_SA_MAIN::~GTA_SA_MAIN() {}

void GTA_SA_MAIN::swichMode(COMPUTE_TYPE type) {
    std::lock_guard<std::mutex> lock(_mtx);
    _currentMode = type;
}

COMPUTE_TYPE GTA_SA_MAIN::getCurrentMode() const {
    return _currentMode;
}

uint64_t GTA_SA_MAIN::getCurrentModeInt() const {
    return static_cast<uint64_t>(_currentMode);
}

void GTA_SA_MAIN::swichMode(uint64_t type) {
    swichMode(static_cast<COMPUTE_TYPE>(type));
}

void GTA_SA_MAIN::setMinRange(std::uint64_t minRange) {
    std::lock_guard<std::mutex> lock(_mtx);
    _minRange = minRange;
}

uint64_t GTA_SA_MAIN::getMinRange() const {
    return _minRange;
}

void GTA_SA_MAIN::setMaxRange(std::uint64_t maxRange) {
    std::lock_guard<std::mutex> lock(_mtx);
    _maxRange = maxRange;
}

uint64_t GTA_SA_MAIN::getMaxRange() const {
    return _maxRange;
}

void GTA_SA_MAIN::setThreadCount(std::uint32_t threadCount) {
    std::lock_guard<std::mutex> lock(_mtx);
    _threadCount = threadCount;
}

uint32_t GTA_SA_MAIN::getThreadCount() const {
    return _threadCount;
}

void GTA_SA_MAIN::setCudaBlockSize(std::uint64_t cudaBlockSize) {
    std::lock_guard<std::mutex> lock(_mtx);
    _cudaBlockSize = cudaBlockSize;
}

uint64_t GTA_SA_MAIN::getCudaBlockSize() const {
    return _cudaBlockSize;
}

void GTA_SA_MAIN::run() {
    std::lock_guard<std::mutex> lock(_mtx);
    std::unique_ptr<GTA_SA_Virtual> gta_sa = constructGTA_SA(_currentMode);
    gta_sa->run();
}

std::unique_ptr<GTA_SA_Virtual> GTA_SA_MAIN::constructGTA_SA(COMPUTE_TYPE type) {
    std::lock_guard<std::mutex> lock(_mtx);
    std::unique_ptr<GTA_SA_Virtual> gta_sa = nullptr;
    switch (type) {
        case COMPUTE_TYPE::STDTHREAD: {
            gta_sa = std::move(std::make_unique<GTA_SA_STDTHREAD>());
            break;
        }
        case COMPUTE_TYPE::OPENMP: {
            gta_sa = std::move(std::make_unique<GTA_SA_OPENMP>());
            break;
        }
        case COMPUTE_TYPE::CUDA: {
#ifdef BUILD_WITH_CUDA
            gta_sa = std::move(std::make_unique<GTA_SA_CUDA>());
            break;
#else
            std::cout << "CUDA not supported, switching to STDTHREAD" << std::endl;
            gta_sa = std::move(std::make_unique<GTA_SA_STDTHREAD>());
            break;
#endif
        }
        case COMPUTE_TYPE::OPENCL: {
#ifdef BUILD_WITH_OPENCL
            gta_sa = std::move(std::make_unique<GTA_SA_OPENCL>());
            break;
#else
            std::cout << "OPENCL not supported, switching to STDTHREAD" << std::endl;
            gta_sa = std::move(std::make_unique<GTA_SA_STDTHREAD>());
            break;
#endif
        }
        default: {
            std::cout << "Unknown calc mode: " << static_cast<uint32_t>(type) << std::endl;
            gta_sa = std::move(std::make_unique<GTA_SA_STDTHREAD>());
            break;
        }
    }

    if (gta_sa == nullptr) {
        std::cout << "Error, gtaSA == nullptr" << std::endl;
        return nullptr;
    }

    gta_sa->setMinRange(_minRange);
    gta_sa->setMaxRange(_maxRange);
    gta_sa->setThreadCount(_threadCount);
    gta_sa->setCudaBlockSize(_cudaBlockSize);

    return gta_sa;
}
