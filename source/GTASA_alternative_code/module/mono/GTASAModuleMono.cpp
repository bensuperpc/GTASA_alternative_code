#include "GTASAModuleMono.hpp"

#include <algorithm> // std::find
#include <cstring> // strlen

GTASAModuleMono::GTASAModuleMono():
    GTASAModule(COMPUTE_TYPE::MONO) {
}

GTASAModuleMono::~GTASAModuleMono() {}

auto GTASAModuleMono::run(std::uint64_t startRange, std::uint64_t endRange) -> std::vector<GTASAResult> {
    _runningInstance++;

    std::vector<GTASAResult> results = {};

    for (std::uint64_t i = startRange; i < endRange; ++i) {
        GTASAResult&& result = this->runner(i);
        if (!result.code.empty()) {
            results.push_back(result);
        }
    }

    std::sort(results.begin(), results.end(), [](const GTASAResult& a, const GTASAResult& b) { return a.index < b.index; });

    _runningInstance--;

    return results;
}

GTASAResult GTASAModuleMono::runner(const std::uint64_t i) const {
    std::array<char, 29> tmp = {0};
    this->generateString(tmp.data(), i);
    const uint32_t crc = this->jamcrc(tmp.data());
    const auto it = std::find(std::begin(GTASAModuleMono::cheatList), std::end(GTASAModuleMono::cheatList), crc);

    // If crc is present in Array
    if (it != std::end(GTASAModuleMono::cheatList)) {
        std::reverse(tmp.data(),
                     tmp.data() + strlen(tmp.data()));  // Invert char array
        
        const uint64_t index = static_cast<uint64_t>(it - std::begin(GTASAModuleMono::cheatList));
        return GTASAResult(i, std::string(tmp.data()), crc, index);                                                                      
    }
    return GTASAResult(i, "", 0, 0);
}
