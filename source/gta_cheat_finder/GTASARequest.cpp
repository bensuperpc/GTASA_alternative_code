#include "GTASARequest.hpp"
#include "module/GTASAModuleThreadpool.hpp"
#include "module/GTASAModuleOpenMP.hpp"

GTASARequest::GTASARequest(std::uint64_t startRange, std::uint64_t endRange, COMPUTE_TYPE type) :
    _startRange(startRange), _endRange(endRange), _type(type), _status(RequestStatus::IDLE) {}

void GTASARequest::start() {
    if (isRunning() || isFinished() || isError()) {
        std::cerr << "Request already running or finished." << std::endl;
        return;
    }

    _future = std::async(std::launch::async, &GTASARequest::run, this);
}

void GTASARequest::run() {
    {
        std::unique_lock<std::shared_mutex> lock(_mutex);
        _status = RequestStatus::RUNNING;
    }
    switch (_type) {
        case COMPUTE_TYPE::STDTHREAD: {
            break;
        }
        case COMPUTE_TYPE::OPENMP: {
            break;
        }
        case COMPUTE_TYPE::CUDA: {
            break;
        }
        case COMPUTE_TYPE::OPENCL: {
            break;
        }
        case COMPUTE_TYPE::NONE: {
            std::cerr << "Unknown calc mode: " << static_cast<uint32_t>(_type) << std::endl;
            _status = RequestStatus::ERROR;
            break;
        }
    }
    {
        std::unique_lock<std::shared_mutex> lock(_mutex);
        _status = RequestStatus::FINISHED;
    }
}

bool GTASARequest::isRunning() const {
    std::shared_lock<std::shared_mutex> lock(_mutex);
    return _status == RequestStatus::RUNNING;
}

bool GTASARequest::isFinished() const {
    std::shared_lock<std::shared_mutex> lock(_mutex);
    return _status == RequestStatus::FINISHED;
}

bool GTASARequest::isError() const {
    std::shared_lock<std::shared_mutex> lock(_mutex);
    return _status == RequestStatus::ERROR;
}

std::uint64_t GTASARequest::getStartRange() const {
    return _startRange;
}

std::uint64_t GTASARequest::getEndRange() const {
    return _endRange;
}

COMPUTE_TYPE GTASARequest::getType() const {
    return _type;
}

std::vector<GTASAResult>& GTASARequest::getResults() {
    return _results;
}

GTASARequest::~GTASARequest() {}