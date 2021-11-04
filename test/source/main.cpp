#include <array>
#include <string>
#include <string_view>
#include <vector>

#include <benchmark/benchmark.h>

static std::array<uint32_t, 256> crc32_lookup = {0};
static constexpr uint32_t polynomial = 0xEDB88320;

void precompute_crc()
{
  crc32_lookup[0] = 0;
  // compute each power of two (all numbers with exactly one bit set)
  uint32_t crc = crc32_lookup[0x80] = polynomial;
  for (std::uint32_t next = 0x40; next != 0; next >>= 1) {
    crc = (crc >> 1) ^ ((crc & 1) * polynomial);
    crc32_lookup[next] = crc;
  }

  for (std::uint32_t power_of_two = 2; power_of_two <= 0x80; power_of_two <<= 1)
  {
    uint32_t crc_extra_bit = crc32_lookup[power_of_two];
    for (std::uint32_t i = 1; i < power_of_two; i++) {
      crc32_lookup[i + power_of_two] = crc32_lookup[i] ^ crc_extra_bit;
    }
  }
}

auto jamcrc(std::string_view my_string) -> std::uint32_t
{
  auto crc = static_cast<uint32_t>(-1);
  auto* current = reinterpret_cast<const unsigned char*>(my_string.data());
  uint64_t length = my_string.length();
  // process eight bytes at once
  while (static_cast<bool>(length--)) {
    crc = (crc >> 8) ^ crc32_lookup[(crc & 0xFF) ^ *current++];
  }
  return crc;
}

static void jamcrc_bench(benchmark::State& state)
{
  // Code inside this loop is measured repeatedly
  std::string created_string("e");
  precompute_crc();

  for (auto _ : state) {
    jamcrc(created_string);
    // Make sure the variable is not optimized away by compiler
    benchmark::DoNotOptimize(created_string);
  }
}
// Register the function as a benchmark
BENCHMARK(jamcrc_bench);

static uint32_t Crc32Lookup[8][256];

void precompute_crc_8()
{
  // same as before
  for (unsigned int i = 0; i <= 0xFF; i++) {
    uint32_t crc = i;
    for (unsigned int j = 0; j < 8; j++)
      crc = (crc >> 1) ^ ((crc & 1) * polynomial);
    Crc32Lookup[0][i] = crc;
  }
  for (unsigned int i = 0; i <= 0xFF; i++) {
    // for Slicing-by-4 and Slicing-by-8
    Crc32Lookup[1][i] =
        (Crc32Lookup[0][i] >> 8) ^ Crc32Lookup[0][Crc32Lookup[0][i] & 0xFF];
    Crc32Lookup[2][i] =
        (Crc32Lookup[1][i] >> 8) ^ Crc32Lookup[0][Crc32Lookup[1][i] & 0xFF];
    Crc32Lookup[3][i] =
        (Crc32Lookup[2][i] >> 8) ^ Crc32Lookup[0][Crc32Lookup[2][i] & 0xFF];
    // only Slicing-by-8
    Crc32Lookup[4][i] =
        (Crc32Lookup[3][i] >> 8) ^ Crc32Lookup[0][Crc32Lookup[3][i] & 0xFF];
    Crc32Lookup[5][i] =
        (Crc32Lookup[4][i] >> 8) ^ Crc32Lookup[0][Crc32Lookup[4][i] & 0xFF];
    Crc32Lookup[6][i] =
        (Crc32Lookup[5][i] >> 8) ^ Crc32Lookup[0][Crc32Lookup[5][i] & 0xFF];
    Crc32Lookup[7][i] =
        (Crc32Lookup[6][i] >> 8) ^ Crc32Lookup[0][Crc32Lookup[6][i] & 0xFF];
  }
}

auto jamcrc_8(std::string_view my_string) -> std::uint32_t
{
  auto crc = static_cast<uint32_t>(-1);
  auto* current = reinterpret_cast<const unsigned char*>(my_string.data());
  uint64_t length = my_string.length();
  // process eight bytes at once
  while (length >= 8) {
    uint32_t one = *current++ ^ crc;
    uint32_t two = *current++;
    crc = Crc32Lookup[7][one & 0xFF] ^ Crc32Lookup[6][(one >> 8) & 0xFF]
        ^ Crc32Lookup[5][(one >> 16) & 0xFF] ^ Crc32Lookup[4][one >> 24]
        ^ Crc32Lookup[3][two & 0xFF] ^ Crc32Lookup[2][(two >> 8) & 0xFF]
        ^ Crc32Lookup[1][(two >> 16) & 0xFF] ^ Crc32Lookup[0][two >> 24];
    length -= 8;
  }
  unsigned char* currentChar = const_cast<unsigned char*>(
      reinterpret_cast<const unsigned char*>(current));
  // remaining 1 to 7 bytes
  while (length--)
    crc = (crc >> 8) ^ Crc32Lookup[0][(crc & 0xFF) ^ *currentChar++];
  return crc;
}

static void jamcrc_8_bench(benchmark::State& state)
{
  // Code inside this loop is measured repeatedly
  std::string created_string("e");
  precompute_crc_8();

  for (auto _ : state) {
    jamcrc_8(created_string);
    // Make sure the variable is not optimized away by compiler
    benchmark::DoNotOptimize(created_string);
  }
}
// Register the function as a benchmark
BENCHMARK(jamcrc_8_bench);

// Run the benchmark
BENCHMARK_MAIN();