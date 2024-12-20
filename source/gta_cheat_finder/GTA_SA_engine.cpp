#include "GTA_SA_engine.hpp"

GTA_SA_ENGINE::GTA_SA_ENGINE() {
    _threadCount = GTA_SA_Virtual::maxThreadSupport();
    _cudaBlockSize = 64;
    swichMode(COMPUTE_TYPE::STDTHREAD);
}

GTA_SA_ENGINE::~GTA_SA_ENGINE() {}

void GTA_SA_ENGINE::swichMode(COMPUTE_TYPE type) {
    _currentGTA_SA.reset();
    switch (type) {
        case COMPUTE_TYPE::STDTHREAD: {
            _currentGTA_SA = std::move(std::make_unique<GTA_SA_STDTHREAD>());
            std::cout << "Using STDTHREAD" << std::endl;
            break;
        }
        case COMPUTE_TYPE::OPENMP: {
            _currentGTA_SA = std::move(std::make_unique<GTA_SA_OPENMP>());
            std::cout << "Using OPENMP" << std::endl;
            break;
        }
        case COMPUTE_TYPE::CUDA: {
#ifdef BUILD_WITH_CUDA
            _currentGTA_SA = std::move(std::make_unique<GTA_SA_CUDA>());
            std::cout << "Using CUDA" << std::endl;
            break;
#else
            std::cout << "CUDA not supported, switching to STDTHREAD" << std::endl;
            _currentGTA_SA = std::move(std::make_unique<GTA_SA_STDTHREAD>());
            break;
#endif
        }
        case COMPUTE_TYPE::OPENCL: {
#ifdef BUILD_WITH_OPENCL
            _currentGTA_SA = std::move(std::make_unique<GTA_SA_OPENCL>());
            break;
#else
            std::cout << "OPENCL not supported, switching to STDTHREAD" << std::endl;
            _currentGTA_SA = std::move(std::make_unique<GTA_SA_STDTHREAD>());
            std::cout << "Using STDTHREAD" << std::endl;
            break;
#endif
        }
        default: {
            std::cout << "Unknown calc mode: " << static_cast<uint32_t>(type) << std::endl;
            _currentGTA_SA = std::move(std::make_unique<GTA_SA_STDTHREAD>());
            break;
        }
    }
    _currentGTA_SA->setMinRange(_minRange);
    _currentGTA_SA->setMaxRange(_maxRange);
    _currentGTA_SA->setThreadCount(_threadCount);
#ifdef BUILD_WITH_CUDA
    _currentGTA_SA->setCudaBlockSize(_cudaBlockSize);
#endif
}

COMPUTE_TYPE GTA_SA_ENGINE::getCurrentMode() const {
    return _currentGTA_SA->type();
}

uint64_t GTA_SA_ENGINE::getCurrentModeInt() const {
    return static_cast<uint64_t>(_currentGTA_SA->type());
}

void GTA_SA_ENGINE::swichMode(uint64_t type) {
    swichMode(static_cast<COMPUTE_TYPE>(type));
}

void GTA_SA_ENGINE::setMinRange(std::uint64_t minRange) {
    
    _minRange = minRange;
}

uint64_t GTA_SA_ENGINE::getMinRange() const {
    return _minRange;
}

void GTA_SA_ENGINE::setMaxRange(std::uint64_t maxRange) {
    
    _maxRange = maxRange;
}

uint64_t GTA_SA_ENGINE::getMaxRange() const {
    return _maxRange;
}

void GTA_SA_ENGINE::setThreadCount(std::uint32_t threadCount) {
    
    _threadCount = threadCount;
}

uint32_t GTA_SA_ENGINE::getThreadCount() const {
    return _threadCount;
}

void GTA_SA_ENGINE::setCudaBlockSize(std::uint64_t cudaBlockSize) {
    
    _cudaBlockSize = cudaBlockSize;
}

uint64_t GTA_SA_ENGINE::getCudaBlockSize() const {
    return _cudaBlockSize;
}

void GTA_SA_ENGINE::run() {
    std::cout << "Running GTA_SA_ENGINE" << std::endl;
    std::lock_guard<std::mutex> lock(_mtx);
    _currentGTA_SA->run();

    std::cout << "GTA_SA_ENGINE finished" << std::endl;
}
