#ifndef GTASAENGINEV2_HPP
#define GTASAENGINEV2_HPP

#include <cstdint>  // std::uint64_t, std::uint32_t
#include <memory>   // std::unique_ptr
#include <mutex>    // std::mutex
#include <string>   // std::string
#include <vector>   // std::vector

#include "GTASARequest.hpp"

#include "module/GTASAModuleVirtual.hpp"

class GTASAEngine {
   public:
    explicit GTASAEngine();
    ~GTASAEngine();
    GTASARequest* addRequest(COMPUTE_TYPE type, std::uint64_t startRange, std::uint64_t endRange);

   private:
      GTASAModuleVirtual * _gtaSAModuleTheadpool = nullptr;
      GTASAModuleVirtual * _gtaSAModuleOpenMP = nullptr;
      GTASAModuleVirtual * _gtaSAModuleCUDA = nullptr;
      GTASAModuleVirtual * _gtaSAModuleOpenCL = nullptr;

      mutable std::shared_mutex _mutex = std::shared_mutex();
      std::vector<std::unique_ptr<GTASARequest>> _requests = {};

};

#endif  // GTA_SA_MAIN_HPP
