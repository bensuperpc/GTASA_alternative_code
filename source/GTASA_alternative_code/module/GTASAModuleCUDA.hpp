#ifndef GTASAMODULEOPENMP_HPP
#define GTASAMODULEOPENMP_HPP

#include "GTASAModule.hpp"

#if __has_include("cuda.h") || __has_include("cuda_runtime.h")
#ifndef BUILD_WITH_CUDA
#define BUILD_WITH_CUDA
#endif
#else
#if _MSC_VER && !__INTEL_COMPILER
#pragma message("CUDA disabled.")
#else
#warning CUDA disabled.
#endif
#endif

#if defined(BUILD_WITH_CUDA)
#include "cuda/wrapper.hpp"
#endif

class GTASAModuleCUDA final : public GTASAModule {
   public:
    explicit GTASAModuleCUDA();
    ~GTASAModuleCUDA();

    std::vector<GTASAResult> run(std::uint64_t startRange, std::uint64_t endRange) override;

    private:
     uint64_t _cudaBlockSize = 64;
};

#endif  // GTASAMODULEOPENMP_HPP
