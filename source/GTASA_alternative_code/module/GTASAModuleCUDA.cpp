#include "GTASAModuleCUDA.hpp"

#include <algorithm> // std::find
#include <cstring> // strlen

GTASAModuleCUDA::GTASAModuleCUDA():
    GTASAModule(COMPUTE_TYPE::CUDA) {
}

GTASAModuleCUDA::~GTASAModuleCUDA() {}

std::vector<GTASAResult> GTASAModuleCUDA::run(std::uint64_t startRange, std::uint64_t endRange)  {
    _runningInstance++;

    std::vector<uint32_t> jamcrc_results = {};
    std::vector<uint64_t> index_results = {};

    std::vector<GTASAResult> results = {};

    my::cuda::launchKernel(jamcrc_results, index_results, startRange, endRange, _cudaBlockSize);

    for (uint64_t i = 0; i < jamcrc_results.size(); ++i) {
        std::array<char, 29> tmpCUDA = {0};

        this->generateString(tmpCUDA.data(), index_results[i]);
        std::reverse(tmpCUDA.data(),
                     tmpCUDA.data() + std::strlen(tmpCUDA.data()));

        const auto&& it = std::find(std::begin(GTASAModuleCUDA::cheatList), std::end(GTASAModuleCUDA::cheatList), jamcrc_results[i]);
        const uint64_t index = static_cast<uint64_t>(it - std::begin(GTASAModuleCUDA::cheatList));
        
        results.emplace_back(index_results[i], std::string(tmpCUDA.data()), jamcrc_results[i], index);                                                                                       
    }

    std::sort(results.begin(), results.end(), [](const GTASAResult& a, const GTASAResult& b) { return a.index < b.index; });

    _runningInstance--;
    return results;
}
