#include <array>
#include <string>
#include <string_view>
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

static void find_string_inv_bench(benchmark::State& state) {
    auto range = state.range(0);

    const auto array_size = 29;
    std::array<char, array_size> tmp = {0};
    GTA_SA_STDTHREAD gtaSA = GTA_SA_STDTHREAD();
    benchmark::DoNotOptimize(tmp);
    benchmark::DoNotOptimize(gtaSA);
    for (auto _ : state) {
        gtaSA.generateString(tmp.data(), range);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * 1);
    state.SetBytesProcessed(state.iterations() * 1 * sizeof(uint64_t));
}

BENCHMARK(find_string_inv_bench)
    ->Name("find_string_inv_bench")
    ->RangeMultiplier(multiplier)
    ->Range(minRange, maxRange)
    ->ThreadRange(minThreadRange, maxThreadRange)
    ->Unit(benchmark::kNanosecond)
    ->Setup(DoSetup)
    ->Teardown(DoTeardown)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Repetitions(repetitions);

// Run the benchmark
// BENCHMARK_MAIN();

/*
template<typename T>
static void BasicBench(benchmark::State& state) {
  const auto range = state.range(0);
  std::string str;
  benchmark::DoNotOptimize(str);

  for (auto _ : state) {
    str = std::to_string(range);
    benchmark::ClobberMemory();
    //state.PauseTiming();
    //state.SkipWithError("No path found");
    //state.ResumeTiming();
  }
  // state.counters["range"] = range;
  state.SetItemsProcessed(state.iterations() * range * range);
  state.SetBytesProcessed(state.iterations() * range * range * sizeof(uint64_t));
}
BENCHMARK(BasicBench<uint64_t>)
    ->Name("BasicBench")
    ->RangeMultiplier(4)
    ->Range(16, 256)
    ->ThreadRange(1, 1)
    ->Unit(benchmark::kNanosecond)
    ->Setup(DoSetup)
    ->Teardown(DoTeardown)
    ->MeasureProcessCPUTime()
    ->UseRealTime();

int main(int argc, char** argv) {
   ::benchmark::Initialize(&argc, argv);
   ::benchmark::RunSpecifiedBenchmarks();
}
*/

int main(int argc, char** argv) {
    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();
}
