#include "GTASAModule.hpp"

GTASAModule::GTASAModule(COMPUTE_TYPE type) : _type(type) {}

GTASAModule::~GTASAModule() {}

GTASAModule::COMPUTE_TYPE GTASAModule::stringToComputeType(std::string_view type) noexcept {
    if (type == "STDTHREAD" || type == "THREADPOOL") {
        return COMPUTE_TYPE::STDTHREAD;
    } else if (type == "OPENMP") {
        return COMPUTE_TYPE::OPENMP;
    } else if (type == "CUDA") {
        return COMPUTE_TYPE::CUDA;
    } else if (type == "OPENCL") {
        return COMPUTE_TYPE::OPENCL;
    }
    return COMPUTE_TYPE::NONE;
}

GTASAModule::COMPUTE_TYPE GTASAModule::type() const {
    return _type;
}

std::uint64_t GTASAModule::runningInstance() const {
    return _runningInstance;
}

auto GTASAModule::jamcrc(std::string_view my_string, const uint32_t previousCrc32) const noexcept -> std::uint32_t {
    auto crc = ~previousCrc32;
    const uint8_t* current = reinterpret_cast<const uint8_t*>(my_string.data());
    uint64_t length = my_string.length();
    // process eight bytes at once
    while (static_cast<bool>(length--)) {
        crc = (crc >> 8) ^ crc32LookupTable[(crc & 0xFF) ^ *current++];
    }
    return crc;
}

void GTASAModule::generateString(char* array, uint64_t n) const noexcept {
    std::uint64_t i = 0;
    while (n) {
        array[i] = alpha[(--n) % 26];
        n /= 26;
        ++i;
    }
}

void GTASAModule::generateStringV2(char* array, uint64_t n) const noexcept {
    std::uint64_t i = 0;
    do {
        array[i++] = alpha[--n % 26];
        n /= 26;
    } while (n > 0);
}
