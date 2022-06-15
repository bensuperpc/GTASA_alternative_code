#include <array>
#include <string>
#include <string_view>
#include <vector>

#include <benchmark/benchmark.h>

#include "GTA_SA_cheat_finder.hpp"

static void find_string_inv_bench(benchmark::State& state)
{
  // Code inside this loop is measured repeatedly
  auto range = state.range(0);

  const auto array_size = 29;
  std::array<char, array_size> tmp = {0};

  for (auto _ : state) {
    GTA_SA::find_string_inv(tmp.data(), range);
    benchmark::DoNotOptimize(tmp);
    // benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations());
  // state.SetBytesProcessed(state.iterations() * state.range(0) *
  // sizeof(char));
}
// Register the function as a benchmark
BENCHMARK(find_string_inv_bench)->Name("find_string_inv_bench")->RangeMultiplier(100)->Range(1, 1000000000000000);

// Run the benchmark
// BENCHMARK_MAIN();
