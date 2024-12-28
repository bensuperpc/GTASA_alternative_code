#ifndef GTASAMODULEOPENMP_HPP
#define GTASAMODULEOPENMP_HPP

#include <module/GTASAModule.hpp>

#if __has_include("CL/cl.h") || __has_include("CL/cl.hpp")
#ifndef BUILD_WITH_OPENCL
#define BUILD_WITH_OPENCL
#endif
#else
#if _MSC_VER && !__INTEL_COMPILER
#pragma message("OpenCL disabled.")
#else
#warning OpenCL disabled.
#endif
#endif

#if defined(BUILD_WITH_OPENCL)
#include "opencl/wrapper.hpp"
#endif

class GTASAModuleOpenCL final : public GTASAModule {
   public:
    explicit GTASAModuleOpenCL();
    ~GTASAModuleOpenCL();

    std::vector<GTASAResult> run(std::uint64_t startRange, std::uint64_t endRange) override;

    private:
     uint64_t _openCLBlockSize = 64;
};

#endif  // GTASAMODULEOPENMP_HPP
