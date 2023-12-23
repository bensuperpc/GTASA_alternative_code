#include "GTA_SA_cheat_finder_cuda.hpp"

GTA_SA_CUDA::GTA_SA_CUDA() {}

GTA_SA_CUDA::~GTA_SA_CUDA() {}

GTA_SA_CUDA& GTA_SA_CUDA::operator=(const GTA_SA_CUDA& other) {
    if (this != &other) {
        this->min_range = other.min_range;
        this->max_range = other.max_range;
        this->num_thread = other.num_thread;
        this->cuda_block_size = other.cuda_block_size;
    }
    return *this;
}

void GTA_SA_CUDA::run() {
    std::cout << "Running with CUDA mode" << std::endl;

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

    IsRunning = true;

    std::array<char, 29> tmp1 = {0};
    std::array<char, 29> tmp2 = {0};

    results.reserve((max_range - min_range) / 20000000 + 1);

    std::cout << "Number of calculations: " << (max_range - min_range) << std::endl;

    this->find_string_inv(tmp1.data(), min_range);
    this->find_string_inv(tmp2.data(), max_range);
    std::cout << "From: " << tmp1.data() << " to: " << tmp2.data() << " Alphabetic sequence" << std::endl;

    std::cout << "Rinimum range: " << std::dec << min_range << std::endl;
    std::cout << "Maximum range: " << std::dec << max_range << std::endl;
    std::cout << "Calculation range: " << std::dec << (max_range - min_range) << std::endl;

    if ((max_range - min_range) < cuda_block_size) {
        std::cout << "Number of calculations is less than cuda_block_size" << std::endl;
    }    

    begin_time = std::chrono::high_resolution_clock::now();
    runner(0);
    end_time = std::chrono::high_resolution_clock::now();

    std::sort(results.begin(), results.end(), [](const result& a, const result& b) { return a.index < b.index; });

    printResult();
    IsRunning = false;
}

void GTA_SA_CUDA::runner(const std::uint64_t) {
    std::vector<uint32_t> jamcrc_results;
    std::vector<uint64_t> index_results;

    my::cuda::launch_kernel(jamcrc_results, index_results, min_range, max_range, cuda_block_size);

    for (uint64_t i = 0; i < jamcrc_results.size(); ++i) {
        std::array<char, 29> tmpCUDA = {0};

        this->find_string_inv(tmpCUDA.data(), index_results[i]);
        std::reverse(tmpCUDA.data(),
                     tmpCUDA.data() + std::strlen(tmpCUDA.data()));  // Invert char array

        const auto&& it = std::find(std::begin(GTA_SA_CUDA::cheatList), std::end(GTA_SA_CUDA::cheatList), jamcrc_results[i]);

        const uint64_t index = static_cast<uint64_t>(it - std::begin(GTA_SA_CUDA::cheatList));
        results.emplace_back(index_results[i], std::string(tmpCUDA.data()), jamcrc_results[i],
                             GTA_SA_Virtual::cheat_list_name.at(static_cast<std::size_t>(index)));  // Save result: calculation position,
                                                                                                    // Alphabetic sequence, CRC, Cheat name
    }
}
