#include "GTASARequest.hpp"

GTASARequest::GTASARequest(GTASAModule* module, std::uint64_t startRange, std::uint64_t endRange) 
    : _startRange(startRange), _endRange(endRange), _status(Status::IDLE), _module(module) {
        if (_module == nullptr) {
            std::unique_lock<std::shared_mutex> lock(_mutex);
            _status = Status::FINISHED;
        }
    }

void GTASARequest::start() {
    if (_status == Status::RUNNING || _status == Status::FINISHED || _status == Status::ERROR) {
        std::cerr << "Request already running or finished." << std::endl;
        return;
    }

    if (_module == nullptr) {
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

    _results = _module->run(_startRange, _endRange);

    {
        std::unique_lock<std::shared_mutex> lock(_mutex);
        _status = Status::FINISHED;
    }
}

GTASARequest::Status GTASARequest::getStatus() const {
    std::shared_lock<std::shared_mutex> lock(_mutex);
    return _status;
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