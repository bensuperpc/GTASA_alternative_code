#ifndef GTASAENGINEV2_HPP
#define GTASAENGINEV2_HPP

#include <cstdint>  // std::uint64_t, std::uint32_t
#include <memory>   // std::unique_ptr
#include <mutex>    // std::mutex
#include <string>   // std::string
#include <vector>   // std::vector

#include "GTASARequest.hpp"

#include "module/GTASAModule.hpp"

class GTASAEngine {
   public:
    explicit GTASAEngine();
    ~GTASAEngine();
    GTASARequest* addRequest(GTASAModule::COMPUTE_TYPE type, std::uint64_t startRange, std::uint64_t endRange);
    GTASARequest* addRequest(std::string&& type, std::uint64_t startRange, std::uint64_t endRange);

    std::vector<std::unique_ptr<GTASARequest>>& getRequests();
    std::shared_mutex& getMutex();
    bool allRequestsFinished() const;
    void waitAllRequests() const;
    GTASAModule* getModule(GTASAModule::COMPUTE_TYPE type) const noexcept;

   private:
      std::unique_ptr<GTASAModule> _gtaSAModuleTheadpool = nullptr;
      std::unique_ptr<GTASAModule> _gtaSAModuleOpenMP = nullptr;
      std::unique_ptr<GTASAModule> _gtaSAModuleCUDA = nullptr;
      std::unique_ptr<GTASAModule> _gtaSAModuleOpenCL = nullptr;

      mutable std::shared_mutex _mutex = std::shared_mutex();
      std::vector<std::unique_ptr<GTASARequest>> _requests = {};

};

#endif  // GTA_SA_MAIN_HPP
