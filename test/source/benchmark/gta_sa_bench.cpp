#include <array>
#include <string>
#include <string_view>
#include <vector>

#include <benchmark/benchmark.h>

#include "GTA_SA_cheat_finder.hpp"

static void find_string_inv_bench(benchmark::State& state) {
    // Code inside this loop is measured repeatedly
    auto range = state.range(0);

    GTA_SA gtaSA;
    gtaSA.min_range = 0;
    gtaSA.max_range = 0x1000;
    gtaSA.calc_mode = 2;

    gtaSA.run();
    // benchmark::DoNotOptimize(tmp);
    for (auto _ : state) {
        // benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * sizeof(uint64_t));
}
// Register the function as a benchmark
BENCHMARK(find_string_inv_bench)->Name("find_string_inv_bench")->RangeMultiplier(100)->Range(1, 1000000000000000);

// Run the benchmark
// BENCHMARK_MAIN();
