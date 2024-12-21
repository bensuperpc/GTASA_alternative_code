#include "GTASAModuleOpenMP.hpp"

GTASAModuleOpenMP::GTASAModuleOpenMP():
    GTASAModuleVirtual(COMPUTE_TYPE::OPENMP) {
}

GTASAModuleOpenMP::~GTASAModuleOpenMP() {}

std::vector<GTASAResult> GTASAModuleOpenMP::run(std::uint64_t startRange, std::uint64_t endRange)  {
    _runningInstance++;

    std::vector<GTASAResult> results = {};

    std::uint64_t i = 0;
#pragma omp parallel for schedule(auto) shared(results)
    for (i = startRange; i <= endRange; i++) {
        GTASAResult&& result = runner(i);
        if (!result.code.empty()) {
            results.push_back(result);
        }
    }

    std::sort(results.begin(), results.end(), [](const GTASAResult& a, const GTASAResult& b) { return a.index < b.index; });

    _runningInstance--;
    return results;
}

GTASAResult GTASAModuleOpenMP::runner(const std::uint64_t i) {
    std::array<char, 29> tmp = {0};
    this->generateString(tmp.data(), i);
    const uint32_t crc = this->jamcrc(tmp.data());
    const auto it = std::find(std::begin(GTASAModuleOpenMP::cheatList), std::end(GTASAModuleOpenMP::cheatList), crc);

    // If crc is present in Array
    if (it != std::end(GTASAModuleOpenMP::cheatList)) {
        std::reverse(tmp.data(),
                     tmp.data() + strlen(tmp.data()));  // Invert char array
        
        const uint64_t index = static_cast<uint64_t>(it - std::begin(GTASAModuleOpenMP::cheatList));
        return GTASAResult(i, std::string(tmp.data()), crc, index);                                                                      
    }
    return GTASAResult(i, "", 0, 0);
}
