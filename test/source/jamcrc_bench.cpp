#include <array>
#include <memory>
#include <string>
#include <vector>

#include <benchmark/benchmark.h>

#include "GTA_SA_cheat_finder.hpp"

#if defined(BUILD_WITH_CUDA)
#  include "cuda/wrapper.hpp"
#endif

// Source: https://github.com/stbrumme/crc32
static std::unique_ptr<char[]> generate_string(const std::uint64_t length)
{
  uint32_t randomNumber = 0x27121978;
  // initialize
  std::unique_ptr<char[]> data(new char[length]);
  for (size_t i = 0; i < length; i++) {
    data[i] = char(randomNumber & 0xFF);
    // simple LCG, see
    // http://en.wikipedia.org/wiki/Linear_congruential_generator
    randomNumber = 1664525 * randomNumber + 1013904223;
  }
  return data;
}

static char* generate_array(const std::uint64_t length)
{
  uint32_t randomNumber = 0x27121978;
  // initialize
  char* data = new char[length];
  for (size_t i = 0; i < length; i++) {
    data[i] = char(randomNumber & 0xFF);
    // simple LCG, see
    // http://en.wikipedia.org/wiki/Linear_congruential_generator
    randomNumber = 1664525 * randomNumber + 1013904223;
  }
  return data;
}

static void jamcrc_bench(benchmark::State& state)
{
  // Code inside this loop is measured repeatedly
  auto size = state.range(0);
  auto c_str = generate_string(size);

  std::string str;
  std::reverse(str.begin(), str.end());
  for (auto i = 0; i < size; i++) {
    str.push_back(c_str[i] + '0');
  }

  for (auto _ : state) {
    benchmark::DoNotOptimize(str);
    GTA_SA::jamcrc(str);
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * state.range(0) * sizeof(char));

  // auto c = c_str.release();
  // delete[] c;
}
BENCHMARK(jamcrc_bench)->Name("jamcrc_bench")->RangeMultiplier(16)->Range(1, 1048576);


static void cuda_jamcrc_bench(benchmark::State& state)
{
  // Code inside this loop is measured repeatedly
  auto size = state.range(0);
  auto c_str = generate_array(size);

  for (auto _ : state) {
    benchmark::DoNotOptimize(c_str);
    my::cuda::jamcrc(c_str, size, 0, 64);
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * state.range(0) * sizeof(char));
}
BENCHMARK(cuda_jamcrc_bench)->Name("cuda_jamcrc_bench")->RangeMultiplier(16)->Range(256, 1048576);
