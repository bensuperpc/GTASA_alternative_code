#include "GTA_SA_cheat_finder_stdthread.hpp"

GTA_SA_STDTHREAD::GTA_SA_STDTHREAD() {}

GTA_SA_STDTHREAD::~GTA_SA_STDTHREAD() {}

GTA_SA_STDTHREAD& GTA_SA_STDTHREAD::operator=(const GTA_SA_STDTHREAD& other) {
    if (this != &other) {
        this->minRange = other.minRange;
        this->maxRange = other.maxRange;
        this->threadCount = other.threadCount;
        this->cudaBlockSize = other.cudaBlockSize;
    }
    return *this;
}

void GTA_SA_STDTHREAD::run() {
    std::cout << "Running with std::thread mode" << std::endl;

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
    std::cout << "Number of calculations: " << (maxRange - minRange) << std::endl;

    IsRunning = true;

    std::array<char, 29> tmp1 = {0};
    std::array<char, 29> tmp2 = {0};

    results.reserve((maxRange - minRange) / 20'000'000 + 1);

    this->generateString(tmp1.data(), minRange);
    this->generateString(tmp2.data(), maxRange);
    std::cout << "From: " << tmp1.data() << " to: " << tmp2.data() << " Alphabetic sequence" << std::endl;
    beginTime = std::chrono::high_resolution_clock::now();

    BS::thread_pool pool(threadCount);

    auto future = pool.submit_blocks(minRange, maxRange, [&](const std::uint64_t& _min_range, const std::uint64_t& _max_range) {
        for (std::uint64_t i = _min_range; i <= _max_range; i++) {
            runner(i);
        }
    });

    /*
    auto future = pool.submit_loop(minRange, maxRange, [&](const std::uint64_t& id) {
        runner(id);
    });
    */

    future.wait();
    endTime = std::chrono::high_resolution_clock::now();

    std::sort(results.begin(), results.end(), [](const GTASAResult& a, const GTASAResult& b) { return a.index < b.index; });

    printResult();
    IsRunning = false;
}

void GTA_SA_STDTHREAD::runner(const std::uint64_t i) {
    std::array<char, 29> tmp = {0};
    this->generateString(tmp.data(), i);            // Generate Alphabetic sequence from uint64_t
                                                    // value, A=1, Z=27, AA = 28, AB = 29
    const uint32_t crc = this->jamcrc(tmp.data());  // JAMCRC
    const auto it = std::find(std::begin(GTA_SA_STDTHREAD::cheatList), std::end(GTA_SA_STDTHREAD::cheatList), crc);

    // If crc is present in Array
    if (it != std::end(GTA_SA_STDTHREAD::cheatList)) {
        std::reverse(tmp.data(),
                     tmp.data() + strlen(tmp.data()));  // Invert char array

        const uint64_t index = static_cast<uint64_t>(it - std::begin(GTA_SA_STDTHREAD::cheatList));
        results.emplace_back(i, std::string(tmp.data()), crc,
                             GTA_SA_Virtual::cheatListName.at(static_cast<std::size_t>(index)));  // Save result: calculation position,
                                                                                                  // Alphabetic sequence, CRC, Cheat name
    }
}

COMPUTE_TYPE GTA_SA_STDTHREAD::type() const {
    return COMPUTE_TYPE::STDTHREAD;
}
