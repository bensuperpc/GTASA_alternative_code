//////////////////////////////////////////////////////////////
//   ____                                                   //
//  | __ )  ___ _ __  ___ _   _ _ __   ___ _ __ _ __   ___  //
//  |  _ \ / _ \ '_ \/ __| | | | '_ \ / _ \ '__| '_ \ / __| //
//  | |_) |  __/ | | \__ \ |_| | |_) |  __/ |  | |_) | (__  //
//  |____/ \___|_| |_|___/\__,_| .__/ \___|_|  | .__/ \___| //
//                             |_|             |_|          //
//////////////////////////////////////////////////////////////
//                                                          //
//  BenLib, 2021                                            //
//  Created: 21, March, 2021                                //
//  Modified: 29, April, 2022                               //
//  file: kernel.cu                                         //
//  Crypto                                                  //
//  Source:
//  https://stackoverflow.com/questions/13553015/cuda-c-linker-error-undefined-reference
//  //
//          https://www.olcf.ornl.gov/tutorials/cuda-vector-addition/ //
//          https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#asynchronous-transfers-and-overlapping-transfers-with-computation__concurrent-copy-and-execute
//          https://www.ce.jhu.edu/dalrymple/classes/602/Class12.pdf //
//          https://create.stephan-brumme.com/crc32/
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

#include "kernel.cuh"

__global__ void jamcrcKernelWrapper(const void* data, uint32_t* result, const uint64_t length, const uint32_t previousCrc32) {
    const uint64_t blockId = blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;
    const uint64_t threadsPerBlock = blockDim.x * blockDim.y * blockDim.z;
    uint64_t id = blockId * threadsPerBlock + threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

    if (id == 0) {
        *result = jamcrc1Byte(data, length, previousCrc32);
    }
}

__global__ void FindAlternativeCheatKernel(uint32_t* crc_result,
                                           uint64_t* index_result,
                                           uint64_t array_size,
                                           uint32_t* arrayIndex,
                                           uint64_t a,
                                           uint64_t b) {
    const uint64_t blockId = blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;
    const uint64_t threadsPerBlock = blockDim.x * blockDim.y * blockDim.z;
    uint64_t id = blockId * threadsPerBlock + threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

    a = id + a;

    if (a <= b) {
        //  Allocate memory for the array
        uint8_t array[29] = {0};

        uint64_t size = 0;
        // Generate the array from index (a)
        GenerateStringKernel(array, a, &size);

        // Calculate the JAMCRC
        const uint32_t result = jamcrc1Byte(array, size, 0);
        // printf("id: %llu, size: %llu, array: %s, crc: 0x%x\n", id, size, array, result);

        bool found = false;
        for (uint8_t i = 0; i < 87; i += 3) {
            if (result == cheatList[i] || result == cheatList[i + 1] || result == cheatList[i + 2]) {
                found = true;
                break;
            }
        }

        if (!found) {
            return;
        }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
        uint32_t localArrayIndex = atomicAdd(arrayIndex, 1);
#else
        uint32_t localArrayIndex = atomicAdd_system(arrayIndex, 1);
#endif

        if (localArrayIndex >= array_size) {
            return;
        }
        crc_result[localArrayIndex] = result;
        index_result[localArrayIndex] = a;
    }
    //__syncthreads();
}

__device__ void GenerateStringKernel(uint8_t* array, uint64_t n, uint64_t* terminatorIndex) {
    // If n < 27
    if (n < 26) {
        array[0] = alpha[n];
        *terminatorIndex = 1;
        return;
    }
    // If n > 27
    uint64_t i = 0;
    while (n > 0) {
        array[i] = alpha[(--n) % 26];
        n /= 26;
        ++i;
    }
    *terminatorIndex = i;
}
