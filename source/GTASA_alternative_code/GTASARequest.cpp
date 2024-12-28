#include "GTASARequest.hpp"

GTASARequest::GTASARequest(GTASAModule* module, std::uint64_t startRange, std::uint64_t endRange) 
    : _startRange(startRange), _endRange(endRange), _status(Status::IDLE), _module(module) {
        if (_module == nullptr) {
            std::unique_lock<std::shared_mutex> lock(_mutex);
            _status = Status::FINISHED;
        }
    }

void GTASARequest::start() {
    if (isRunning() || isFinished() || isError()) {
        std::cerr << "Request already running or finished." << std::endl;
        return;
    }

    std::cout << "Starting request." << std::endl;

    _future = std::async(std::launch::async, &GTASARequest::run, this);
}

void GTASARequest::run() {
    {
        std::unique_lock<std::shared_mutex> lock(_mutex);
        _status = Status::RUNNING;
    }

    if (_module != nullptr) {
        _results = _module->run(_startRange, _endRange);
    }

    {
        std::unique_lock<std::shared_mutex> lock(_mutex);
        _status = Status::FINISHED;
    }
}

bool GTASARequest::isRunning() const {
    std::shared_lock<std::shared_mutex> lock(_mutex);
    return _status == Status::RUNNING;
}

bool GTASARequest::isFinished() const {
    std::shared_lock<std::shared_mutex> lock(_mutex);
    return _status == Status::FINISHED || _status == Status::ERROR;
}

bool GTASARequest::isError() const {
    std::shared_lock<std::shared_mutex> lock(_mutex);
    return _status == Status::ERROR;
}

bool GTASARequest::isStarted() const {
    std::shared_lock<std::shared_mutex> lock(_mutex);
    return _status != Status::IDLE;
}

std::uint64_t GTASARequest::getStartRange() const {
    return _startRange;
}

std::uint64_t GTASARequest::getEndRange() const {
    return _endRange;
}

GTASAModule::COMPUTE_TYPE GTASARequest::getType() const {
    if (_module == nullptr) {
        return GTASAModule::COMPUTE_TYPE::NONE;
    }

    return _module->type();
}

std::vector<GTASAResult>& GTASARequest::getResults() {
    return _results;
}

GTASARequest::~GTASARequest() {}