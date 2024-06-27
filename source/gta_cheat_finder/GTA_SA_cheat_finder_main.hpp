#ifndef GTA_SA_MAIN_HPP
#define GTA_SA_MAIN_HPP

#include <cstdint>  // std::uint64_t, std::uint32_t
#include <memory>   // std::unique_ptr
#include <mutex>    // std::mutex
#include <string>   // std::string
#include <vector>   // std::vector

#include "GTA_SA_cheat_finder_virtual.hpp"

#ifdef BUILD_WITH_CUDA
#include "GTA_SA_cheat_finder_cuda.hpp"
#endif  // BUILD_WITH_CUDA

#ifdef BUILD_WITH_OPENCL
#include "GTA_SA_cheat_finder_opencl.hpp"
#endif  // BUILD_WITH_OPENCL

#include "GTA_SA_cheat_finder_openmp.hpp"
#include "GTA_SA_cheat_finder_stdthread.hpp"

class GTA_SA_MAIN {
   public:
    explicit GTA_SA_MAIN();
    ~GTA_SA_MAIN();

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
    std::unique_ptr<GTA_SA_Virtual> constructGTA_SA(COMPUTE_TYPE type);
    COMPUTE_TYPE _currentMode = COMPUTE_TYPE::STDTHREAD;
    //std::vector<std::unique_ptr<GTA_SA_Virtual>> _gTA_SA_list;
    uint64_t _minRange = 0;
    uint64_t _maxRange = 0;
    uint32_t _threadCount;
    uint64_t _cudaBlockSize;

    std::mutex _mtx;
};

#endif  // GTA_SA_MAIN_HPP
