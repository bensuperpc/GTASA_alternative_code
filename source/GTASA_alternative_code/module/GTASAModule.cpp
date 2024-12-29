#include <algorithm> // std::find
#include <array>    // std::array
#include <cstring> // strlen

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

auto GTASAModule::type() const -> COMPUTE_TYPE {
    return _type;
}

auto GTASAModule::runningInstance() const -> std::uint64_t {
    return _runningInstance.load();
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

auto GTASAModule::runner(const std::uint64_t i) const -> GTASAResult {
    std::array<char, 29> tmp = {0};
    this->generateString(tmp.data(), i);
    const uint32_t crc = this->jamcrc(tmp.data());
    const auto it = std::find(std::begin(GTASAModule::cheatList), std::end(GTASAModule::cheatList), crc);

    // If crc is present in Array
    if (it != std::end(GTASAModule::cheatList)) {
        std::reverse(tmp.data(),
                     tmp.data() + strlen(tmp.data()));  // Invert char array
        
        const uint64_t index = static_cast<uint64_t>(it - std::begin(GTASAModule::cheatList));
        return GTASAResult(i, std::string(tmp.data()), crc, index);                                                                      
    }
    return GTASAResult(i, "", 0, 0);
}
