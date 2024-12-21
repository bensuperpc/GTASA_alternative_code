#ifndef GTASAMODULEVIRTUAL_HPP
#define GTASAMODULEVIRTUAL_HPP

#include <cstdint>  // std::uint64_t
#include <vector>   // std::vector
#include <atomic>   // std::atomic
#include "GTASAResult.hpp"

enum class COMPUTE_TYPE { NONE, STDTHREAD, OPENMP, CUDA, OPENCL };

class GTASAModuleVirtual {
   public:
    virtual ~GTASAModuleVirtual();
    virtual std::vector<GTASAResult> run(std::uint64_t startRange, std::uint64_t endRange) = 0;
    COMPUTE_TYPE type() const;
    std::uint64_t runningInstance() const;
   protected:
    explicit GTASAModuleVirtual(COMPUTE_TYPE type);

    const COMPUTE_TYPE _type = COMPUTE_TYPE::NONE;

    std::atomic<std::uint64_t> _runningInstance = 0;
};

#endif  // GTASAMODULEVIRTUAL_HPP
