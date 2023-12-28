#ifndef OPENCL_WRAPPER_HPP
#define OPENCL_WRAPPER_HPP

#include <vector>
#include <cstdint>

#if !defined(CL_TARGET_OPENCL_VERSION)
#define CL_TARGET_OPENCL_VERSION 300
//#define CL_HPP_TARGET_OPENCL_VERSION 300
//#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#endif

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

namespace my::opencl
{
    uint32_t jamcrc(const void* data, const uint64_t length, const uint32_t previousCrc32);
    void launchKernel(std::vector<uint32_t>& jamcrc_results,
                                std::vector<uint64_t>& index_results,
                                const uint64_t minRange,
                                const uint64_t maxRange,
                                const uint64_t cudaBlockSize);
}

#endif  // OPENCL_WRAPPER_HPP
