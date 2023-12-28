#include "GTA_SA_cheat_finder_virtual.hpp"

GTA_SA_Virtual::GTA_SA_Virtual() {}

GTA_SA_Virtual::~GTA_SA_Virtual() {}

/**
 * @brief To get JAMCRC with boost libs
 * @param my_string String input
 * @return uint32_t with JAMCRC value
 */
auto GTA_SA_Virtual::jamcrc(std::string_view my_string, const uint32_t previousCrc32) const noexcept -> std::uint32_t {
    auto crc = ~previousCrc32;
    const uint8_t* current = reinterpret_cast<const uint8_t*>(my_string.data());
    uint64_t length = my_string.length();
    // process eight bytes at once
    while (static_cast<bool>(length--)) {
        crc = (crc >> 8) ^ crc32LookupTable[(crc & 0xFF) ^ *current++];
    }
    return crc;
}

/**
 * \brief Generate Alphabetic sequence from uint64_t value, A=1, Z=27, AA =
 * 28, AB = 29 \param n index in base 26 \param array return array
 */
void GTA_SA_Virtual::generateString(char* array, uint64_t n) const noexcept {
    // If n < 27
    if (n < 26) [[unlikely]] {
        array[0] = alpha[static_cast<std::size_t>(n)];
        return;
    }
    // If n > 27
    std::uint64_t i = 0;
    while (n) {
        array[i] = alpha[(--n) % 26];
        n /= 26;
        ++i;
    }
}

void GTA_SA_Virtual::clear() {
    results.clear();
}

uint32_t GTA_SA_Virtual::maxThreadSupport() {
#if defined(_OPENMP)
    return static_cast<uint32_t>(omp_get_max_threads());
#else
    return static_cast<uint32_t>(std::thread::hardware_concurrency());
#endif
}

void GTA_SA_Virtual::printResult() const {
    std::cout << "" << std::endl;

    constexpr auto display_val = 18;

    std::cout << std::setw(display_val + 4) << "Iter. N°" << std::setw(display_val + 3) << "Code" << std::setw(display_val + 11)
              << "JAMCRC value" << std::setw(display_val + 16) << "Associated code" << std::endl;

    for (auto& result : results) {
        std::cout << std::setw(display_val + 2) << std::dec << result.index << std::setw(display_val + 5) << result.code
                  << std::setw(display_val) << "0x" << std::hex << result.jamcrc << std::setw(display_val + 20) << result.associated_code
                  << std::endl;
    }
    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::duration<double>>(endTime - beginTime).count() << " sec"
              << std::endl;  // Display time

    std::cout << "This program execute: " << std::fixed
              << (static_cast<double>(maxRange - minRange) /
                  std::chrono::duration_cast<std::chrono::duration<double>>(endTime - beginTime).count()) /
                     1000000
              << " MOps/sec" << std::endl;  // Display perf
    std::cout << "" << std::endl;
    std::cout << "Number of results: " << std::dec << results.size() << std::endl;
}
