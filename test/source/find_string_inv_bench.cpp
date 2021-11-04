#include <array>
#include <string>
#include <string_view>
#include <vector>

#include <benchmark/benchmark.h>

#include "gta_sa_lib.hpp"

static void find_string_inv_bench(benchmark::State& state)
{
  // Code inside this loop is measured repeatedly
  auto range = state.range(0);

  const auto array_size = 29;
  std::array<char, array_size> tmp1 = {0};

  for (auto _ : state) {
    gta::find_string_inv(tmp1.data(), range);
    // Make sure the variable is not optimized away by compiler
    // benchmark::DoNotOptimize(tmp1);
  }
}
// Register the function as a benchmark
BENCHMARK(find_string_inv_bench)->Arg(10);
BENCHMARK(find_string_inv_bench)->Arg(100);
BENCHMARK(find_string_inv_bench)->Arg(1000);
BENCHMARK(find_string_inv_bench)->Arg(1000000);
BENCHMARK(find_string_inv_bench)->Arg(1000000000);

// Run the benchmark
BENCHMARK_MAIN();
