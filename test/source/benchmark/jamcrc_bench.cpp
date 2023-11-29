#include <algorithm>
#include <array>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <benchmark/benchmark.h>

#include "GTA_SA_cheat_finder_openmp.hpp"
#include "GTA_SA_cheat_finder_stdthread.hpp"
#if defined(BUILD_WITH_CUDA)
#include "GTA_SA_cheat_finder_cuda.hpp"
#endif  // BUILD_WITH_CUDA

static constexpr int64_t multiplier = 16;
static constexpr int64_t minRange = 1;
static constexpr int64_t maxRange = 1048576;
static constexpr int64_t minThreadRange = 1;
static constexpr int64_t maxThreadRange = 1;
static constexpr int64_t repetitions = 1;

static void DoSetup([[maybe_unused]] const benchmark::State& state) {}

static void DoTeardown([[maybe_unused]] const benchmark::State& state) {}

// https://stackoverflow.com/questions/440133/how-do-i-create-a-random-alpha-numeric-string-in-c/444614#444614
template <typename T = std::mt19937_64>
auto random_generator() -> T {
    auto constexpr seed_bytes = sizeof(typename T::result_type) * T::state_size;
    auto constexpr seed_len = seed_bytes / sizeof(std::seed_seq::result_type);
    auto seed = std::array<std::seed_seq::result_type, seed_len>();
    auto dev = std::random_device();
    std::generate_n(std::begin(seed), seed_len, std::ref(dev));
    auto seed_seq = std::seed_seq(std::begin(seed), std::end(seed));
    return T{seed_seq};
}

std::string generate_string(const size_t size) {
    thread_local auto rng = random_generator<std::mt19937_64>();
    std::uniform_int_distribution<char> distribution(std::numeric_limits<char>::lowest(), std::numeric_limits<char>::max());
    std::string str(size, '\0');
    std::generate_n(str.begin(), size, [&]() { return distribution(rng); });

    return str;
}

static void jamcrc_bench(benchmark::State& state) {
    auto size = state.range(0);
    std::string str = generate_string(size);
    benchmark::DoNotOptimize(str);
    GTA_SA_STDTHREAD gtaSA = GTA_SA_STDTHREAD();
    for (auto _ : state) {
        auto res = gtaSA.jamcrc(str);
        benchmark::DoNotOptimize(res);
        benchmark::ClobberMemory();
        // state.PauseTiming();
        // state.SkipWithError("No path found");
        // state.ResumeTiming();
    }
    state.SetBytesProcessed(state.iterations() * state.range(0) * sizeof(std::string::value_type));
    state.SetItemsProcessed(state.iterations() * state.range(0));
}
BENCHMARK(jamcrc_bench)
    ->Name("jamcrc_bench")
    ->RangeMultiplier(multiplier)
    ->Range(minRange, maxRange)
    ->ThreadRange(minThreadRange, maxThreadRange)
    ->Unit(benchmark::kNanosecond)
    ->Setup(DoSetup)
    ->Teardown(DoTeardown)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Repetitions(repetitions);

/*
#if defined(BUILD_WITH_CUDA)
static void cuda_jamcrc_bench(benchmark::State &state) {
  // Code inside this loop is measured repeatedly
  auto size = state.range(0);
  std::string str = generate_array(size);
  GTA_SA_CUDA gtaSA = GTA_SA_CUDA();
  for (auto _ : state) {
    benchmark::DoNotOptimize(str);
    //gtaSA.jamcrc(str);
    my::cuda::jamcrc(str.c_str(), size, 0, 64);
    benchmark::ClobberMemory();
  }
    state.SetBytesProcessed(state.iterations() * state.range(0) * sizeof(std::string::value_type));
    state.SetItemsProcessed(state.iterations() * state.range(0));
}
BENCHMARK(cuda_jamcrc_bench)
    ->Name("cuda_jamcrc_bench")
    ->RangeMultiplier(multiplier)
    ->Range(minRange, maxRange)
    ->ThreadRange(minThreadRange, maxThreadRange)
    ->Unit(benchmark::kNanosecond)
    ->Setup(DoSetup)
    ->Teardown(DoTeardown)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Repetitions(repetitions);
#endif
*/

int main(int argc, char** argv) {
    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();
}
