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

__host__ uint32_t my::cuda::jamcrc(const void* data, const uint64_t length, const uint32_t previousCrc32, const uint32_t cuda_block_size) {
    int device = 0;
    cudaGetDevice(&device);

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    // Calculate length of the array with max_range and min_range
    uint64_t data_size = (length) * sizeof(char);
    uint32_t* data_cuda = nullptr;

    uint64_t result_size = 1 * sizeof(uint32_t);
    uint32_t* result_cuda = nullptr;

    cudaMallocManaged(&data_cuda, data_size, cudaMemAttachGlobal);
    cudaMallocManaged(&result_cuda, result_size, cudaMemAttachGlobal);

    cudaStreamAttachMemAsync(stream, &data_cuda);
    cudaStreamAttachMemAsync(stream, &result_cuda);

    cudaMemPrefetchAsync(data_cuda, data_size, device, stream);
    cudaMemPrefetchAsync(result_cuda, result_size, device, stream);

    memcpy(data_cuda, data, data_size);
    *result_cuda = 0;

    uint64_t grid_size = static_cast<uint64_t>(ceil(static_cast<double>(1) / cuda_block_size));

    dim3 threads(static_cast<uint32_t>(cuda_block_size), 1, 1);
    dim3 grid(static_cast<uint32_t>(grid_size), 1, 1);

    jamcrcKernelWrapper<<<grid, threads, device, stream>>>(data_cuda, result_cuda, length, previousCrc32);

    cudaStreamSynchronize(stream);
    cudaDeviceSynchronize();

    // std::cout << "result_cuda: " << *result_cuda << std::endl;

    cudaFree(data_cuda);
    cudaStreamDestroy(stream);

    return *result_cuda;
}

__host__ void my::cuda::launch_kernel(std::vector<uint32_t>& jamcrc_results,
                                      std::vector<uint64_t>& index_results,
                                      const uint64_t min_range,
                                      const uint64_t max_range,
                                      const uint64_t cuda_block_size) {
    std::cout << "Launching kernel..." << std::endl;

    uint64_t calcRange = max_range - min_range;

    // int device = -1;
    // cudaGetDevice(&device);

    int device = 0;
    cudaGetDevice(&device);

    /*
    int priority_high, priority_low;
    cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
    cudaStream_t st_high, st_low;
    cudaStreamCreateWithPriority(&st_high, cudaStreamNonBlocking, priority_high);
    cudaStreamCreateWithPriority(&st_low, cudaStreamNonBlocking, priority_low);
    */

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    // cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128 * 1024 * 1024);

    // Calculate length of the array with max_range and min_range (Estimate size of the array)
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

    uint64_t rest = static_cast<uint64_t>((calcRange / cuda_block_size) + (calcRange % cuda_block_size));
    uint32_t cubeRoot = static_cast<uint32_t>(std::ceil(std::cbrt(static_cast<long double>(rest))));

    dim3 grid = dim3(0, 0, 0);

    if (rest < 2'000'000'000ULL) [[likely]] {
        grid.x = rest;
        grid.y = 1;
        grid.z = 1;
    } else {
        grid.x = cubeRoot;
        grid.y = cubeRoot;
        grid.z = cubeRoot;
    }

    uint64_t total_grid_size = grid.x * grid.y * grid.z;

    dim3 threads = dim3(0, 0, 0);
    threads.x = cuda_block_size;
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

    FindAlternativeCheatKernel<<<grid, threads, device, stream>>>(jamcrcResultsPtr, indexResultsPtr, arrayLength, ResultsSize, min_range,
                                                                  max_range);

    cudaStreamSynchronize(stream);

    jamcrc_results.insert(jamcrc_results.end(), jamcrcResultsPtr, jamcrcResultsPtr + *ResultsSize);
    index_results.insert(index_results.end(), indexResultsPtr, indexResultsPtr + *ResultsSize);

    cudaDeviceSynchronize();
    cudaFree(jamcrcResultsPtr);
    cudaFree(indexResultsPtr);
    cudaFree(ResultsSize);

    cudaStreamDestroy(stream);
    // cudaStreamDestroy(st_high);
    // cudaStreamDestroy(st_low);

    std::cout << "CUDA Kernel finished" << std::endl;
}
