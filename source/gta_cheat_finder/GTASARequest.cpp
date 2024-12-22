#include "GTASARequest.hpp"
#include "module/GTASAModuleThreadpool.hpp"
#include "module/GTASAModuleOpenMP.hpp"

GTASARequest::GTASARequest(GTASAModuleVirtual& module, std::uint64_t startRange, std::uint64_t endRange) 
    : _startRange(startRange), _endRange(endRange), _status(RequestStatus::IDLE), _module(module) {}

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

bool GTASARequest::isStarted() const {
    std::shared_lock<std::shared_mutex> lock(_mutex);
    return _status != RequestStatus::IDLE;
}

std::uint64_t GTASARequest::getStartRange() const {
    return _startRange;
}

std::uint64_t GTASARequest::getEndRange() const {
    return _endRange;
}

COMPUTE_TYPE GTASARequest::getType() const {
    return _module.type();
}

std::vector<GTASAResult>& GTASARequest::getResults() {
    return _results;
}

GTASARequest::~GTASARequest() {}