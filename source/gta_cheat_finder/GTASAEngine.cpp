#include "GTASAEngine.hpp"

#ifdef BUILD_WITH_CUDA
#include "module/GTASAModuleCUDA.hpp"
#endif  // BUILD_WITH_CUDA

#ifdef BUILD_WITH_OPENCL
#include "module/GTASAModuleOpenCL.hpp"
#endif  // BUILD_WITH_OPENCL

#include "module/GTASAModuleThreadpool.hpp"
#include "module/GTASAModuleOpenMP.hpp"

GTASAEngine::GTASAEngine() {
    _gtaSAModuleTheadpool = new GTASAModuleThreadpool();
    _gtaSAModuleOpenMP = new GTASAModuleOpenMP();

#ifdef BUILD_WITH_CUDA
    _gtaSAModuleCUDA = new GTASAModuleCUDA();
#endif  // BUILD_WITH_CUDA

#ifdef BUILD_WITH_OPENCL
    _gtaSAModuleOpenCL = new GTASAModuleOpenCL();
#endif  // BUILD_WITH_OPENCL
}

GTASARequest* GTASAEngine::addRequest(COMPUTE_TYPE type, std::uint64_t startRange, std::uint64_t endRange) {
    switch (type) {
        case COMPUTE_TYPE::STDTHREAD: {
            std::unique_ptr<GTASARequest> request = std::make_unique<GTASARequest>(*_gtaSAModuleTheadpool, startRange, endRange);
            request->start();
            std::unique_lock<std::shared_mutex> lock(_mutex);
            _requests.push_back(std::move(request));
            return _requests.back().get();
            break;
        }
        case COMPUTE_TYPE::OPENMP: {
            std::unique_ptr<GTASARequest> request = std::make_unique<GTASARequest>(*_gtaSAModuleOpenMP, startRange, endRange);
            request->start();
            std::unique_lock<std::shared_mutex> lock(_mutex);
            _requests.push_back(std::move(request));
            return _requests.back().get();
            break;
        }
        case COMPUTE_TYPE::CUDA: {
#ifdef BUILD_WITH_CUDA
            std::unique_ptr<GTASARequest> request = std::make_unique<GTASARequest>(*_gtaSAModuleCUDA, startRange, endRange);
            request->start();
            std::unique_lock<std::shared_mutex> lock(_mutex);
            _requests.push_back(std::move(request));
            return _requests.back().get();
#else
            std::cerr << "CUDA not supported." << std::endl;
            return nullptr;
#endif
            break;
        }
        case COMPUTE_TYPE::OPENCL: {
#ifdef BUILD_WITH_OPENCL
            std::unique_ptr<GTASARequest> request = std::make_unique<GTASARequest>(*_gtaSAModuleOpenCL, startRange, endRange);
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
        case COMPUTE_TYPE::NONE: {
            std::cerr << "Unknown calc mode: " << static_cast<uint32_t>(type) << std::endl;
            return nullptr;
            break;
        }
    }
    return nullptr;
}

GTASAEngine::~GTASAEngine() {}

