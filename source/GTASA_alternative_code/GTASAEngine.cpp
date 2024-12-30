#include "GTASAEngine.hpp"

#include "module/mono/GTASAModuleMono.hpp"

#if ENABLE_OPENCL == 1
#include "module/opencl/GTASAModuleOpenCL.hpp"
#endif  // BUILD_WITH_OPENCL

#if ENABLE_CUDA == 1
#include "module/cuda/GTASAModuleCUDA.hpp"
#endif  // ENABLE_CUDA

#if ENABLE_THREADPOOL == 1
#include "module/threadpool/GTASAModuleThreadpool.hpp"
#endif  // ENABLE_THREADPOOL

#if ENABLE_OPENMP == 1
#include "module/openmp/GTASAModuleOpenMP.hpp"
#endif  // ENABLE_OPENMP

GTASAEngine::GTASAEngine() {
    _gtaSAModuleMono = std::make_unique<GTASAModuleMono>();

#if ENABLE_THREADPOOL == 1
    _gtaSAModuleTheadpool = std::make_unique<GTASAModuleThreadpool>();
#endif  // ENABLE_THREADPOOL

#if ENABLE_OPENMP == 1
    _gtaSAModuleOpenMP = std::make_unique<GTASAModuleOpenMP>();
#endif  // ENABLE_OPENMP

#if ENABLE_CUDA == 1
    _gtaSAModuleCUDA = std::make_unique<GTASAModuleCUDA>();
#endif  // ENABLE_CUDA

#if ENABLE_OPENCL == 1
    _gtaSAModuleOpenCL = std::make_unique<GTASAModuleOpenCL>();
#endif  // BUILD_WITH_OPENCL
}

GTASAEngine::~GTASAEngine() {}

GTASARequest* GTASAEngine::addRequest(std::string&& type, std::uint64_t startRange, std::uint64_t endRange) {
    return addRequest(GTASAModule::stringToComputeType(type), startRange, endRange);
}

GTASARequest* GTASAEngine::addRequest(GTASAModule::COMPUTE_TYPE type, std::uint64_t startRange, std::uint64_t endRange) {
    switch (type) {
        case GTASAModule::COMPUTE_TYPE::MONO: {
            std::unique_ptr<GTASARequest> request = std::make_unique<GTASARequest>(_gtaSAModuleMono.get(), startRange, endRange);
            request->start();
            std::unique_lock<std::shared_mutex> lock(_mutex);
            _requests.push_back(std::move(request));
            return _requests.back().get();
            break;
        }
        case GTASAModule::COMPUTE_TYPE::STDTHREAD: {
#if ENABLE_THREADPOOL == 1
            std::unique_ptr<GTASARequest> request = std::make_unique<GTASARequest>(_gtaSAModuleTheadpool.get(), startRange, endRange);
#else
            std::cerr << "STDTHREAD not supported, falling back to Mono module" << std::endl;
            std::unique_ptr<GTASARequest> request = std::make_unique<GTASARequest>(_gtaSAModuleMono.get(), startRange, endRange);
#endif  // ENABLE_THREADPOOL
            request->start();
            std::unique_lock<std::shared_mutex> lock(_mutex);
            _requests.push_back(std::move(request));
            return _requests.back().get();
            break;
        }
        case GTASAModule::COMPUTE_TYPE::OPENMP: {
#if ENABLE_OPENMP == 1
            std::unique_ptr<GTASARequest> request = std::make_unique<GTASARequest>(_gtaSAModuleOpenMP.get(), startRange, endRange);
#else
            std::cerr << "OPENMP not supported, falling back to Mono module" << std::endl;
            std::unique_ptr<GTASARequest> request = std::make_unique<GTASARequest>(_gtaSAModuleMono.get(), startRange, endRange);
#endif  // ENABLE_OPENMP
            request->start();
            std::unique_lock<std::shared_mutex> lock(_mutex);
            _requests.push_back(std::move(request));
            return _requests.back().get();
            break;
        }
        case GTASAModule::COMPUTE_TYPE::CUDA: {
#if ENABLE_CUDA == 1
            std::unique_ptr<GTASARequest> request = std::make_unique<GTASARequest>(_gtaSAModuleCUDA.get(), startRange, endRange);
#else
            std::cerr << "CUDA not supported, falling back to Mono module" << std::endl;
            std::unique_ptr<GTASARequest> request = std::make_unique<GTASARequest>(_gtaSAModuleMono.get(), startRange, endRange);
#endif  // ENABLE_CUDA
            request->start();
            std::unique_lock<std::shared_mutex> lock(_mutex);
            _requests.push_back(std::move(request));
            return _requests.back().get();
            break;
        }
        case GTASAModule::COMPUTE_TYPE::OPENCL: {
#if ENABLE_OPENCL == 1
            std::unique_ptr<GTASARequest> request = std::make_unique<GTASARequest>(_gtaSAModuleOpenCL.get(), startRange, endRange);
#else
            std::cerr << "OPENCL not supported, falling back to Mono module" << std::endl;
            std::unique_ptr<GTASARequest> request = std::make_unique<GTASARequest>(_gtaSAModuleMono.get(), startRange, endRange);
#endif  // BUILD_WITH_OPENCL
            request->start();
            std::unique_lock<std::shared_mutex> lock(_mutex);
            _requests.push_back(std::move(request));
            return _requests.back().get();
            break;
        }
        case GTASAModule::COMPUTE_TYPE::NONE: {
            std::cerr << "Unknown calc mode: " << static_cast<uint32_t>(type) << std::endl;
            return nullptr;
            break;
        }
    }
    return nullptr;
}

std::vector<std::unique_ptr<GTASARequest>>& GTASAEngine::getRequests() {
    return _requests;
}

std::shared_mutex& GTASAEngine::getMutex() {
    return _mutex;
}

bool GTASAEngine::allRequestsFinished() const {
    std::shared_lock<std::shared_mutex> lock(_mutex);
    for (const auto& request : _requests) {
        if (!request->isFinished()) {
            return false;
        }
    }
    return true;
}

void GTASAEngine::waitAllRequests() const {
    while (!allRequestsFinished()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

GTASAModule* GTASAEngine::getModule(GTASAModule::COMPUTE_TYPE type) const noexcept {
    switch (type) {
        case GTASAModule::COMPUTE_TYPE::MONO:
            return nullptr;
            break;
        case GTASAModule::COMPUTE_TYPE::STDTHREAD:
            return _gtaSAModuleTheadpool.get();
            break;
        case GTASAModule::COMPUTE_TYPE::OPENMP:
            return _gtaSAModuleOpenMP.get();
            break;
        case GTASAModule::COMPUTE_TYPE::CUDA:
            return _gtaSAModuleCUDA.get();
            break;
        case GTASAModule::COMPUTE_TYPE::OPENCL:
            return _gtaSAModuleOpenCL.get();
            break;
        case GTASAModule::COMPUTE_TYPE::NONE:
            return nullptr;
            break;
    }
    return nullptr;
}
