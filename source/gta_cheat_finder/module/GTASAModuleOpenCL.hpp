#ifndef GTASAMODULEOPENMP_HPP
#define GTASAMODULEOPENMP_HPP

#include <algorithm>    // std::find
#include <array>        // std::array
#include <chrono>       // std::chrono
#include <cmath>        // std::ceil
#include <cstring>      // strlen
#include <iomanip>      // std::setw
#include <iostream>     // std::cout
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <tuple>        // std::pair
#include <utility>      // std::make_pair
#include <vector>       // std::vector

#include "GTASAModuleVirtual.hpp"

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

class GTASAModuleOpenCL final : public GTASAModuleVirtual {
   public:
    explicit GTASAModuleOpenCL();
    ~GTASAModuleOpenCL();

    std::vector<GTASAResult> run(std::uint64_t startRange, std::uint64_t endRange) override;

    private:
     uint64_t _openCLBlockSize = 64;
};

#endif  // GTASAMODULEOPENMP_HPP
