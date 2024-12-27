#ifndef GTA_SA_MAIN_HPP
#define GTA_SA_MAIN_HPP

#include <cstdint>  // std::uint64_t, std::uint32_t
#include <memory>   // std::unique_ptr
#include <mutex>    // std::mutex
#include <string>   // std::string
#include <vector>   // std::vector

#include "GTASA_alternative_code_virtual.hpp"

#ifdef BUILD_WITH_CUDA
#include "GTASA_alternative_code_cuda.hpp"
#endif  // BUILD_WITH_CUDA

#ifdef BUILD_WITH_OPENCL
#include "GTASA_alternative_code_opencl.hpp"
#endif  // BUILD_WITH_OPENCL

#include "GTASA_alternative_code_openmp.hpp"
#include "GTASA_alternative_code_stdthread.hpp"

class GTA_SA_ENGINE {
   public:
    explicit GTA_SA_ENGINE();
    ~GTA_SA_ENGINE();

    void swichMode(COMPUTE_TYPE type);
    void swichMode(uint64_t type);
    COMPUTE_TYPE getCurrentMode() const;
    uint64_t getCurrentModeInt() const;

    void setMinRange(std::uint64_t minRange);
    uint64_t getMinRange() const;
    void setMaxRange(std::uint64_t maxRange);
    uint64_t getMaxRange() const;
    void setThreadCount(std::uint32_t threadCount);
    uint32_t getThreadCount() const;
    void setCudaBlockSize(std::uint64_t cudaBlockSize);
    uint64_t getCudaBlockSize() const;

    void run();

   private:
    //std::vector<std::unique_ptr<GTA_SA_Virtual>> _gTA_SA_list;
    uint64_t _minRange = 0;
    uint64_t _maxRange = 0;
    uint32_t _threadCount = 1;
    uint64_t _cudaBlockSize = 64;

    std::mutex _mtx;

    std::unique_ptr<GTA_SA_Virtual> _currentGTA_SA = nullptr;
};

#endif  // GTA_SA_MAIN_HPP
