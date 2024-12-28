#include "GTASAEngine.hpp"

#ifdef BUILD_WITH_OPENCL
#include "module/GTASAModuleOpenCL.hpp"
#endif  // BUILD_WITH_OPENCL

#include "module/GTASAModuleCUDA.hpp"
#include "module/GTASAModuleThreadpool.hpp"
#include "module/GTASAModuleOpenMP.hpp"

GTASAEngine::GTASAEngine() {
    _gtaSAModuleTheadpool = std::make_unique<GTASAModuleThreadpool>();
    _gtaSAModuleOpenMP = std::make_unique<GTASAModuleOpenMP>();
    _gtaSAModuleCUDA = std::make_unique<GTASAModuleCUDA>();

#ifdef BUILD_WITH_OPENCL
    _gtaSAModuleOpenCL = std::make_unique<GTASAModuleOpenCL>();
#endif  // BUILD_WITH_OPENCL
}

GTASAEngine::~GTASAEngine() {}

GTASARequest* GTASAEngine::addRequest(std::string&& type, std::uint64_t startRange, std::uint64_t endRange) {
    return addRequest(GTASAModule::stringToComputeType(type), startRange, endRange);
}

GTASARequest* GTASAEngine::addRequest(GTASAModule::COMPUTE_TYPE type, std::uint64_t startRange, std::uint64_t endRange) {
    switch (type) {
        case GTASAModule::COMPUTE_TYPE::STDTHREAD: {
            std::unique_ptr<GTASARequest> request = std::make_unique<GTASARequest>(_gtaSAModuleTheadpool.get(), startRange, endRange);
            request->start();
            std::unique_lock<std::shared_mutex> lock(_mutex);
            _requests.push_back(std::move(request));
            return _requests.back().get();
            break;
        }
        case GTASAModule::COMPUTE_TYPE::OPENMP: {
            std::unique_ptr<GTASARequest> request = std::make_unique<GTASARequest>(_gtaSAModuleOpenMP.get(), startRange, endRange);
            request->start();
            std::unique_lock<std::shared_mutex> lock(_mutex);
            _requests.push_back(std::move(request));
            return _requests.back().get();
            break;
        }
        case GTASAModule::COMPUTE_TYPE::CUDA: {
            std::unique_ptr<GTASARequest> request = std::make_unique<GTASARequest>(_gtaSAModuleCUDA.get(), startRange, endRange);
            request->start();
            std::unique_lock<std::shared_mutex> lock(_mutex);
            _requests.push_back(std::move(request));
            return _requests.back().get();
            break;
        }
        case GTASAModule::COMPUTE_TYPE::OPENCL: {
#ifdef BUILD_WITH_OPENCL
            std::unique_ptr<GTASARequest> request = std::make_unique<GTASARequest>(_gtaSAModuleOpenCL.get(), startRange, endRange);
            request->start();
            std::unique_lock<std::shared_mutex> lock(_mutex);
            _requests.push_back(std::move(request));
            return _requests.back().get();
#else
            std::cerr << "OPENCL not supported." << std::endl;
            return nullptr;
#endif
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
