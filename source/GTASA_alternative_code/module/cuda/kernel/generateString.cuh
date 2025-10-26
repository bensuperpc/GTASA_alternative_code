#ifndef CUDA_GENERATESTRING_CUH
#define CUDA_GENERATESTRING_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__device__ void generateStringKernel(uint8_t* array, uint64_t n, uint64_t* terminatorIndex);
__device__ void generateStringKernelV2(uint8_t* array, uint64_t n, uint64_t* terminatorIndex);

__device__ const uint8_t alpha[27] = {"ABCDEFGHIJKLMNOPQRSTUVWXYZ"};

#endif
