#include <cstdint>

#include "gta_sa_lib.hpp"

/**
 * \brief Source: https://create.stephan-brumme.com/crc32/#slicing-by-8-overview
 */
#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus >= 201703L)
auto gta::jamcrc(std::string_view my_string) -> std::uint32_t
{
#else

#  if _MSC_VER && !__INTEL_COMPILER
#    pragma message( \
        "C++17 is not enabled, the program will be less efficient with previous standards")
#  else
#warning C++17 is not enabled, the program will be less efficient with previous standards.
#  endif

auto gta::jamcrc(const std::string& my_string) -> std::uint32_t
{
#endif
  auto crc = static_cast<uint32_t>(-1);
  auto* current = reinterpret_cast<const unsigned char*>(my_string.data());
  size_t length = my_string.length();
  // process eight bytes at once
  while (static_cast<bool>(length--)) {
    crc = (crc >> 8) ^ crc32_lookup[(crc & 0xFF) ^ *current++];
  }
  return crc;
}

/**
 * \brief Generate Alphabetic sequence from size_t value, A=0, Z=26, AA = 27, BA
 * = 28 = 29 T \param n index in base 26 \param array return array
 */
void gta::find_string_inv(char* array, uint64_t n)
{
  const std::uint32_t string_size_alphabet {alphabet_size + 1};
  const std::array<char, string_size_alphabet> alpha {ALPHABET_UP};
  // If n < 27
  if (n < string_size_alphabet - 1) {
    array[0] = alpha[n];
    return;
  }
  // If n > 27
  std::size_t i = 0;
  while (n > 0) {
    array[i] = alpha[(--n) % alphabet_size];
    n /= alphabet_size;
    ++i;
  }
}

/**
 * \brief Fill crc32_lookup table
 * Source: https://create.stephan-brumme.com/crc32/#slicing-by-8-overview
 */
void gta::precompute_crc()
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