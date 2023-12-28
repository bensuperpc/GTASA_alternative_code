#include "GTA_SA_cheat_finder_openmp.hpp"

GTA_SA_OPENMP::GTA_SA_OPENMP() {}

GTA_SA_OPENMP::~GTA_SA_OPENMP() {}

GTA_SA_OPENMP& GTA_SA_OPENMP::operator=(const GTA_SA_OPENMP& other) {
    if (this != &other) {
        this->min_range = other.min_range;
        this->max_range = other.max_range;
        this->num_thread = other.num_thread;
        this->cuda_block_size = other.cuda_block_size;
    }
    return *this;
}

void GTA_SA_OPENMP::run() {
    std::cout << "Running with OpenMP mode" << std::endl;

    std::cout << "Max thread support: " << GTA_SA_Virtual::max_thread_support() << std::endl;
    std::cout << "Running with: " << num_thread << " threads" << std::endl;

    if (min_range > max_range) {
        std::cout << "Min range value: '" << min_range << "' can't be greater than Max range value: '" << max_range << "'" << std::endl;
        return;
    }

    if ((max_range - min_range) < 1) {
        std::cout << "Search range is too small." << std::endl;
        std::cout << "Min range value: '" << min_range << "' Max range value: '" << max_range << "'" << std::endl;
        return;
    }
    std::cout << "Number of calculations: " << (max_range - min_range) << std::endl;

    IsRunning = true;

    std::array<char, 29> tmp1 = {0};
    std::array<char, 29> tmp2 = {0};

    results.reserve((max_range - min_range) / 20000000 + 1);

    this->find_string_inv(tmp1.data(), min_range);
    this->find_string_inv(tmp2.data(), max_range);
    std::cout << "From: " << tmp1.data() << " to: " << tmp2.data() << " Alphabetic sequence" << std::endl;
    begin_time = std::chrono::high_resolution_clock::now();

    omp_set_num_threads(static_cast<int>(num_thread));

#ifdef _MSC_VER
    static std::int64_t i = 0;  // OpenMP (2.0) on Windows doesn't support unsigned variable
#pragma omp parallel for shared(results) schedule(dynamic)
    for (i = static_cast<std::int64_t>(min_range); i <= static_cast<std::int64_t>(max_range); i++) {
        cpu_runner(static_cast<std::int64_t>(i));
    }
#else
    std::uint64_t i = 0;
#pragma omp parallel for schedule(auto) shared(results)
    for (i = min_range; i <= max_range; i++) {
        runner(i);
    }
#endif

    end_time = std::chrono::high_resolution_clock::now();

    std::sort(results.begin(), results.end(), [](const result& a, const result& b) { return a.index < b.index; });

    printResult();
    IsRunning = false;
}

void GTA_SA_OPENMP::runner(const std::uint64_t i) {
    std::array<char, 29> tmp = {0};
    this->find_string_inv(tmp.data(),
                          i);                       // Generate Alphabetic sequence from uint64_t
                                                    // value, A=1, Z=27, AA = 28, AB = 29
    const uint32_t crc = this->jamcrc(tmp.data());  // JAMCRC
    const auto it = std::find(std::begin(GTA_SA_OPENMP::cheatList), std::end(GTA_SA_OPENMP::cheatList), crc);

    // If crc is present in Array
    if (it != std::end(GTA_SA_OPENMP::cheatList)) {
        std::reverse(tmp.data(),
                     tmp.data() + strlen(tmp.data()));  // Invert char array

        const uint64_t index = static_cast<uint64_t>(it - std::begin(GTA_SA_OPENMP::cheatList));
        results.emplace_back(i, std::string(tmp.data()), crc,
                             GTA_SA_Virtual::cheatListName.at(static_cast<std::size_t>(index)));  // Save result: calculation position,
                                                                                                    // Alphabetic sequence, CRC, Cheat name
    }
}
