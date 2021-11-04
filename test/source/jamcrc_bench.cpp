#include <array>
#include <string>
#include <string_view>
#include <vector>

#include <benchmark/benchmark.h>

#include "gta_sa_lib.hpp"

static void jamcrc_bench(benchmark::State& state)
{
  // Code inside this loop is measured repeatedly
  std::string created_string("01234567890123456789");
  gta::precompute_crc();

  for (auto _ : state) {
    gta::jamcrc(created_string);
    // Make sure the variable is not optimized away by compiler
    benchmark::DoNotOptimize(created_string);
  }
}
// Register the function as a benchmark
BENCHMARK(jamcrc_bench);

// Run the benchmark
BENCHMARK_MAIN();