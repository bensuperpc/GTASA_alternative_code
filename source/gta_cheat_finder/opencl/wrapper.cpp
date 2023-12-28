#include "wrapper.hpp"
#include <iostream>

#include "kernel.hpp"

uint32_t my::opencl::jamcrc(const void* data, const uint64_t length, const uint32_t previousCrc32) {
    if (length == 0) {
        return ~previousCrc32;
    }

    uint64_t data_size = (length) * sizeof(char);
    uint32_t result = 0;

	cl_platform_id platformId = NULL;
	cl_device_id deviceID = NULL;
	cl_uint retNumDevices;
	cl_uint retNumPlatforms;
	cl_int ret = clGetPlatformIDs(1, &platformId, &retNumPlatforms);
    if (ret != CL_SUCCESS) {
        std::cout << "Error: Failed to get platform id!" << std::endl;
        return 0;
    }

	ret = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_DEFAULT, 1, &deviceID, &retNumDevices);
    if (ret != CL_SUCCESS) {
        std::cout << "Error: Failed to get device id!" << std::endl;
        return 0;
    }

    cl_context context = clCreateContext(NULL, 1, &deviceID, NULL, NULL,  &ret);

    cl_command_queue commandQueue = clCreateCommandQueue(context, deviceID, 0, &ret);

    cl_mem openclData = clCreateBuffer(context, CL_MEM_READ_ONLY, data_size, NULL, &ret);
    if (ret != CL_SUCCESS) {
        std::cout << "Error: Failed to allocate device memory: openclData" << std::endl;
        return 0;
    }

    cl_mem openclLength = clCreateBuffer(context, CL_MEM_READ_ONLY, 1 * sizeof(uint64_t), NULL, &ret);
    if (ret != CL_SUCCESS) {
        std::cout << "Error: Failed to allocate device memory: openclLength" << std::endl;
        return 0;
    }

    cl_mem openclPreviousCrc32 = clCreateBuffer(context, CL_MEM_READ_ONLY, 1 * sizeof(uint32_t), NULL, &ret);
    if (ret != CL_SUCCESS) {
        std::cout << "Error: Failed to allocate device memory: openclPreviousCrc32" << std::endl;
        return 0;
    }

    cl_mem openclResult = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 1 * sizeof(uint32_t), NULL, &ret);
    if (ret != CL_SUCCESS) {
        std::cout << "Error: Failed to allocate device memory: openclResult" << std::endl;
        return 0;
    }

    ret = clEnqueueWriteBuffer(commandQueue, openclData, CL_TRUE, 0, data_size, data, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        std::cout << "Error: Failed to write to source array!" << std::endl;
        return 0;
    }

    ret = clEnqueueWriteBuffer(commandQueue, openclLength, CL_TRUE, 0, 1 * sizeof(uint64_t), &length, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        std::cout << "Error: Failed to write to source array!" << std::endl;
        return 0;
    }

    ret = clEnqueueWriteBuffer(commandQueue, openclPreviousCrc32, CL_TRUE, 0, 1 * sizeof(uint32_t), &previousCrc32, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        std::cout << "Error: Failed to write to source array!" << std::endl;
        return 0;
    }

    static std::string kernelSourceStr = my::opencl::kernel::jamcrc_table() + my::opencl::kernel::jamcrc1Byte();

    const char *kernelSource = kernelSourceStr.c_str();
	size_t kernelSize = kernelSourceStr.size();

    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, (const size_t *)&kernelSize, &ret);


    char buildOptions[] = "-cl-std=CL3.0";

    ret = clBuildProgram(program, 1, &deviceID, (const char*)&buildOptions, NULL, NULL);
    if (ret != CL_SUCCESS) {
        std::cout << "Error: Failed to build program!" << std::endl;
        
        size_t len;
        char buffer[8192];
        clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        std::cout << std::string(buffer, len) << std::endl;
        return 0;
    }

	cl_kernel kernel = clCreateKernel(program, "jamcrc1Byte", &ret);

    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), static_cast<void*>(&openclData));
    if (ret != CL_SUCCESS) {
        std::cout << "Error: Failed to set kernel arguments!" << std::endl;
        return 0;
    }
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem),static_cast<void*>(&openclLength));
    if (ret != CL_SUCCESS) {
        std::cout << "Error: Failed to set kernel arguments!" << std::endl;
        return 0;
    }

    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), static_cast<void*>(&openclPreviousCrc32));
    if (ret != CL_SUCCESS) {
        std::cout << "Error: Failed to set kernel arguments!" << std::endl;
        return 0;
    }

    ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), static_cast<void*>(&openclResult));
    if (ret != CL_SUCCESS) {
        std::cout << "Error: Failed to set kernel arguments!" << std::endl;
        return 0;
    }

    size_t globalItemSize = 1;
    size_t localItemSize = 1;
    ret = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &globalItemSize, &localItemSize, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        std::cout << "Error: Failed to execute kernel!" << std::endl;
        return 0;
    }

    ret = clFinish(commandQueue);
    if (ret != CL_SUCCESS) {
        std::cout << "Error: Failed to finish!" << std::endl;
        if (ret == CL_OUT_OF_RESOURCES) {
            std::cout << "CL_OUT_OF_RESOURCES" << std::endl;
        } else if (ret == CL_OUT_OF_HOST_MEMORY) {
            std::cout << "CL_OUT_OF_HOST_MEMORY" << std::endl;
        } else {
            std::cout << "Unknown error: " << ret << std::endl;
        }
        return 0;
    }

    ret = clEnqueueReadBuffer(commandQueue, openclResult, CL_TRUE, 0, 1 * sizeof(uint32_t), &result, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        std::cout << "Error: Failed to read output array!" << std::endl;
        return 0;
    }

	ret = clFlush(commandQueue);
	ret = clReleaseCommandQueue(commandQueue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(openclData);
	ret = clReleaseMemObject(openclLength);
	ret = clReleaseMemObject(openclResult);
    ret = clReleaseMemObject(openclPreviousCrc32);
	ret = clReleaseContext(context);

    return result;
}

void my::opencl::launchKernel(std::vector<uint32_t>& jamcrc_results,
                                std::vector<uint64_t>& index_results,
                                const uint64_t min_range,
                                const uint64_t max_range,
                                const uint64_t cuda_block_size) {
    
    
    }