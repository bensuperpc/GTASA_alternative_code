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

#include "wrapper.hpp"

#include <cstring>

__host__ uint32_t my::cuda::jamcrc(const void* data, const uint64_t length, const uint32_t previousCrc32, const uint32_t cudaBlockSize) {
    int device = 0;
    cudaGetDevice(&device);

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    uint64_t data_size = (length) * sizeof(char);
    uint32_t* cudaData = nullptr;

    uint64_t result_size = 1 * sizeof(uint32_t);
    uint32_t* result_cuda = nullptr;

    cudaMallocManaged(&cudaData, data_size, cudaMemAttachGlobal);
    cudaMallocManaged(&result_cuda, result_size, cudaMemAttachGlobal);

    cudaStreamAttachMemAsync(stream, &cudaData);
    cudaStreamAttachMemAsync(stream, &result_cuda);

    cudaMemPrefetchAsync(cudaData, data_size, device, stream);
    cudaMemPrefetchAsync(result_cuda, result_size, device, stream);

    memcpy(cudaData, data, data_size);
    *result_cuda = 0;

    dim3 threads(1, 1, 1);
    dim3 grid(1, 1, 1);

    jamcrcKernelWrapper<<<grid, threads, device, stream>>>(cudaData, result_cuda, length, previousCrc32);

    cudaStreamSynchronize(stream);
    cudaDeviceSynchronize();

    uint32_t result = *result_cuda;
    cudaFree(cudaData);
    cudaFree(result_cuda);
    cudaStreamDestroy(stream);

    return result;
}

__host__ void my::cuda::launchKernel(std::vector<uint32_t>& jamcrc_results,
                                     std::vector<uint64_t>& index_results,
                                     const uint64_t minRange,
                                     const uint64_t maxRange,
                                     const uint64_t cudaBlockSize) {
    std::cout << "Launching kernel..." << std::endl;

    uint64_t calcRange = maxRange - minRange;

    int device = 0;
    cudaGetDevice(&device);

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    // cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128 * 1024 * 1024);

    // Calculate length of the array with maxRange and minRange (Estimate size of the array)
    uint64_t arrayLength = static_cast<uint64_t>((calcRange / 20'000'000) + 128);
    uint64_t jamcrcResultsSize = arrayLength * sizeof(uint32_t);
    uint64_t indexResultsSize = arrayLength * sizeof(uint64_t);

    uint32_t* jamcrcResultsPtr = nullptr;
    uint64_t* indexResultsPtr = nullptr;

    uint32_t* ResultsSize = nullptr;

    cudaMallocManaged(&jamcrcResultsPtr, jamcrcResultsSize, cudaMemAttachGlobal);
    cudaMallocManaged(&indexResultsPtr, indexResultsSize, cudaMemAttachGlobal);
    cudaMallocManaged(&ResultsSize, 1 * sizeof(uint32_t), cudaMemAttachGlobal);

    cudaStreamAttachMemAsync(stream, &jamcrcResultsPtr);
    cudaStreamAttachMemAsync(stream, &indexResultsSize);
    cudaStreamAttachMemAsync(stream, &ResultsSize);

    cudaMemPrefetchAsync(jamcrcResultsPtr, jamcrcResultsSize, device, stream);
    cudaMemPrefetchAsync(indexResultsPtr, indexResultsSize, device, stream);
    cudaMemPrefetchAsync(ResultsSize, 1 * sizeof(uint32_t), device, stream);

    if (jamcrcResultsPtr == nullptr || indexResultsPtr == nullptr || ResultsSize == nullptr) {
        std::cout << "Error: Could not allocate memory on GPU" << std::endl;
        return;
    }

    uint64_t rest = static_cast<uint64_t>((calcRange / cudaBlockSize) + (calcRange % cudaBlockSize));
    uint32_t cubeRoot = static_cast<uint32_t>(std::ceil(std::cbrt(static_cast<long double>(rest))));

    dim3 grid = dim3(0, 0, 0);

    if (rest < 2'000'000'000ULL) [[likely]] {
        grid.x = static_cast<unsigned int>(rest);
        grid.y = 1;
        grid.z = 1;
    } else {
        grid.x = cubeRoot;
        grid.y = cubeRoot;
        grid.z = cubeRoot;
    }

    uint64_t total_grid_size = grid.x * grid.y * grid.z;

    dim3 threads = dim3(0, 0, 0);
    threads.x = static_cast<unsigned int>(cudaBlockSize);
    threads.y = 1;
    threads.z = 1;
    uint64_t total_cuda_block_size = threads.x * threads.y * threads.z;

    std::cout << "Grid size: " << std::dec << grid.x << "x" << grid.y << "x" << grid.z << " = " << total_grid_size << std::endl;
    std::cout << "Block size: " << std::dec << threads.x << "x" << threads.y << "x" << threads.z << " = " << total_cuda_block_size << std::endl;

    uint64_t total_threads = total_grid_size * total_cuda_block_size;

    if (total_threads < calcRange) {
        std::cout << "Error: Grid size * Block size is smaller than the range to calculate: " << std::dec
                  << grid.x * grid.y * grid.z * threads.x * threads.y * threads.z << " < " << calcRange << std::endl;
        return;
    }

    findAlternativeCheatKernel<<<grid, threads, device, stream>>>(indexResultsPtr, jamcrcResultsPtr, arrayLength, ResultsSize, minRange,
                                                                  maxRange);

    cudaStreamSynchronize(stream);

    jamcrc_results.insert(jamcrc_results.end(), jamcrcResultsPtr, jamcrcResultsPtr + *ResultsSize);
    index_results.insert(index_results.end(), indexResultsPtr, indexResultsPtr + *ResultsSize);

    cudaDeviceSynchronize();
    cudaFree(jamcrcResultsPtr);
    cudaFree(indexResultsPtr);
    cudaFree(ResultsSize);

    cudaStreamDestroy(stream);

    std::cout << "CUDA Kernel finished" << std::endl;
}
