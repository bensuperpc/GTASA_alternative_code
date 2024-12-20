#ifndef GTASAMODULEVIRTUAL_HPP
#define GTASAMODULEVIRTUAL_HPP

#include "GTASAResult.hpp"

enum class COMPUTE_TYPE { NONE, STDTHREAD, OPENMP, CUDA, OPENCL };

class GTASAModuleVirtual {
   public:
    virtual ~GTASAModuleVirtual();
    virtual void run(std::uint64_t startRange, std::uint64_t endRange) = 0;
    COMPUTE_TYPE type() const;
   protected:
    explicit GTASAModuleVirtual(COMPUTE_TYPE type);

    const COMPUTE_TYPE _type = COMPUTE_TYPE::NONE;
};

#endif  // GTASAMODULEVIRTUAL_HPP
