#include "jamcrc.cuh"

__device__ uint32_t jamcrc1Byte(const void* data, uint16_t length, const uint32_t previousCrc32) {
    uint32_t crc = ~previousCrc32;
    uint8_t* current = (uint8_t*)data;
    while (length--)
        crc = (crc >> 8) ^ crc32LookupTable[0][(crc & 0xFF) ^ *current++];
    return crc;
}

__device__ uint32_t jamcrc4Byte(const void* data, uint16_t length, const uint32_t previousCrc32) {
    uint32_t crc = ~previousCrc32;
    uint32_t* current = (uint32_t*)data;
    while (length >= 4) {
        uint32_t one = *current++ ^ crc;
        crc = crc32LookupTable[0][(one >> 24) & 0xFF] ^ crc32LookupTable[1][(one >> 16) & 0xFF] ^ crc32LookupTable[2][(one >> 8) & 0xFF] ^
              crc32LookupTable[3][one & 0xFF];
        length -= 4;
    }

    uint8_t* currentChar = (uint8_t*)(current);
    while (length-- != 0)
        crc = (crc >> 8) ^ crc32LookupTable[0][(crc & 0xFF) ^ *currentChar++];

    return crc;
}
