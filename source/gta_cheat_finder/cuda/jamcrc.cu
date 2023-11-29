#include "jamcrc.cuh"

__device__ uint32_t jamcrcKernel(const void* data, uint64_t length, const uint32_t previousCrc32) {
    uint32_t crc = ~previousCrc32;
    uint8_t* current = (uint8_t*)data;
    while (length--)
        crc = (crc >> 8) ^ crc32_lookup[(crc & 0xFF) ^ *current++];
    return crc;
}