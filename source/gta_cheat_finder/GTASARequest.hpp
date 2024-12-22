#ifndef GTASAREQUEST_HPP
#define GTASAREQUEST_HPP

#include <cstdint>  // std::uint64_t, std::uint32_t
#include <string>   // std::string
#include <vector>   // std::vector
#include <shared_mutex> // std::shared_mutex
#include <mutex> // std::mutex
#include <iostream> // std::cerr
#include <future>
#include <functional>

#include "module/GTASAModuleVirtual.hpp"

class GTASARequest {
   public:
    enum class RequestStatus { IDLE, RUNNING, FINISHED, ERROR };
    explicit GTASARequest(GTASAModuleVirtual& module, std::uint64_t startRange, std::uint64_t endRange);
    ~GTASARequest();

    void start();

    bool isStarted() const;
    bool isRunning() const;
    bool isFinished() const;
    bool isError() const;

    std::uint64_t getStartRange() const;
    std::uint64_t getEndRange() const;
    COMPUTE_TYPE getType() const;
    std::vector<GTASAResult>& getResults();

    private:
     void run();
     std::future<void> _future;


     mutable std::shared_mutex _mutex = std::shared_mutex();

     std::uint64_t _startRange = 0;
     std::uint64_t _endRange = 0;
     RequestStatus _status = RequestStatus::IDLE;
     GTASAModuleVirtual& _module;
     std::vector<GTASAResult> _results = {};
};

#endif  // GTASAREQUEST_HPP
