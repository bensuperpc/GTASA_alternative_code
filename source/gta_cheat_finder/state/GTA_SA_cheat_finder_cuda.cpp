#include "GTA_SA_cheat_finder_cuda.hpp"

GTA_SA_CUDA::GTA_SA_CUDA() {}

GTA_SA_CUDA::~GTA_SA_CUDA() {}

GTA_SA_CUDA& GTA_SA_CUDA::operator=(const GTA_SA_CUDA& other) {
    if (this != &other) {
        this->minRange = other.minRange;
        this->maxRange = other.maxRange;
        this->threadCount = other.threadCount;
        this->cudaBlockSize = other.cudaBlockSize;
    }
    return *this;
}

void GTA_SA_CUDA::run() {
    std::cout << "Running with CUDA mode" << std::endl;

    std::cout << "Max thread support: " << GTA_SA_Virtual::maxThreadSupport() << std::endl;
    std::cout << "Running with: " << threadCount << " threads" << std::endl;

    if (minRange > maxRange) {
        std::cout << "Min range value: '" << minRange << "' can't be greater than Max range value: '" << maxRange << "'" << std::endl;
        return;
    }

    if ((maxRange - minRange) < 1) {
        std::cout << "Search range is too small." << std::endl;
        std::cout << "Min range value: '" << minRange << "' Max range value: '" << maxRange << "'" << std::endl;
        return;
    }

    IsRunning = true;

    std::array<char, 29> tmp1 = {0};
    std::array<char, 29> tmp2 = {0};

    results.reserve((maxRange - minRange) / 20000000 + 1);

    std::cout << "Number of calculations: " << (maxRange - minRange) << std::endl;

    this->generateString(tmp1.data(), minRange);
    this->generateString(tmp2.data(), maxRange);
    std::cout << "From: " << tmp1.data() << " to: " << tmp2.data() << " Alphabetic sequence" << std::endl;

    std::cout << "Rinimum range: " << std::dec << minRange << std::endl;
    std::cout << "Maximum range: " << std::dec << maxRange << std::endl;
    std::cout << "Calculation range: " << std::dec << (maxRange - minRange) << std::endl;

    if ((maxRange - minRange) < cudaBlockSize) {
        std::cout << "Number of calculations is less than cudaBlockSize" << std::endl;
    }

    beginTime = std::chrono::high_resolution_clock::now();
    runner(0);
    endTime = std::chrono::high_resolution_clock::now();

    std::sort(results.begin(), results.end(), [](const result& a, const result& b) { return a.index < b.index; });

    printResult();
    IsRunning = false;
}

void GTA_SA_CUDA::runner(const std::uint64_t) {
    std::vector<uint32_t> jamcrc_results;
    std::vector<uint64_t> index_results;

    my::cuda::launchKernel(jamcrc_results, index_results, minRange, maxRange, cudaBlockSize);

    for (uint64_t i = 0; i < jamcrc_results.size(); ++i) {
        std::array<char, 29> tmpCUDA = {0};

        this->generateString(tmpCUDA.data(), index_results[i]);
        std::reverse(tmpCUDA.data(),
                     tmpCUDA.data() + std::strlen(tmpCUDA.data()));  // Invert char array

        const auto&& it = std::find(std::begin(GTA_SA_CUDA::cheatList), std::end(GTA_SA_CUDA::cheatList), jamcrc_results[i]);

        const uint64_t index = static_cast<uint64_t>(it - std::begin(GTA_SA_CUDA::cheatList));
        results.emplace_back(index_results[i], std::string(tmpCUDA.data()), jamcrc_results[i],
                             GTA_SA_Virtual::cheatListName.at(static_cast<std::size_t>(index)));  // Save result: calculation position,
                                                                                                  // Alphabetic sequence, CRC, Cheat name
    }
}
