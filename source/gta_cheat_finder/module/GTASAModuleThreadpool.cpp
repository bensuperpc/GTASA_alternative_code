#include "GTASAModuleThreadpool.hpp"

#include <algorithm> // std::find
#include <cstring> // strlen

GTASAModuleThreadpool::GTASAModuleThreadpool():
    GTASAModuleVirtual(COMPUTE_TYPE::STDTHREAD),
    _pool(BS::thread_pool<BS::tp::none>(0)) {
}

GTASAModuleThreadpool::~GTASAModuleThreadpool() {}

std::vector<GTASAResult> GTASAModuleThreadpool::run(std::uint64_t startRange, std::uint64_t endRange)  {
    _runningInstance++;

    std::vector<GTASAResult> results = {};

    std::mutex mtx = std::mutex();
    auto loop = [&](const std::uint64_t& id) {
        GTASAResult&& result = runner(id);
        if (!result.code.empty()) {
            std::lock_guard<std::mutex> lock(mtx);
            results.push_back(result);
        }
    };

    auto future = _pool.submit_loop(startRange, endRange, loop, 0);

    future.wait();

    std::sort(results.begin(), results.end(), [](const GTASAResult& a, const GTASAResult& b) { return a.index < b.index; });

    _runningInstance--;

    return results;
}

GTASAResult GTASAModuleThreadpool::runner(const std::uint64_t i) {
    std::array<char, 29> tmp = {0};
    this->generateString(tmp.data(), i);
    const uint32_t crc = this->jamcrc(tmp.data());
    const auto it = std::find(std::begin(GTASAModuleThreadpool::cheatList), std::end(GTASAModuleThreadpool::cheatList), crc);

    // If crc is present in Array
    if (it != std::end(GTASAModuleThreadpool::cheatList)) {
        std::reverse(tmp.data(),
                     tmp.data() + strlen(tmp.data()));  // Invert char array
        
        const uint64_t index = static_cast<uint64_t>(it - std::begin(GTASAModuleThreadpool::cheatList));
        return GTASAResult(i, std::string(tmp.data()), crc, index);                                                                      
    }
    return GTASAResult(i, "", 0, 0);
}
