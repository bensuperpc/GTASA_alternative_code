#include "generateString.cuh"

__device__ void generateStringKernel(uint8_t* array, uint64_t n, uint64_t* terminatorIndex) {
    // If n > 27
    uint64_t i = 0;
    while (n > 0) {
        array[i] = alpha[(--n) % 26];
        n /= 26;
        ++i;
    }
    *terminatorIndex = i;
}

__device__ void generateStringKernelV2(uint8_t* array, uint64_t n, uint64_t* terminatorIndex) {
    uint64_t i = 0;
    do {
        array[i++] = alpha[--n % 26];
        n /= 26;
    } while (n > 0);

    *terminatorIndex = i;
}

