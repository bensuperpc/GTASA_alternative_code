#include <cstdint>

#include "gta_sa_lib.hpp"

/**
 * \brief Source: https://create.stephan-brumme.com/crc32/#slicing-by-8-overview
 */
#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus >= 201703L)
std::uint32_t jamcrc(std::string_view my_string)
{
#else
#warning C++17 is not enabled, the program will be less efficient with previous standards.
auto jamcrc(const std::string& my_string) -> std::uint32_t
{
#endif
  auto crc = static_cast<uint32_t>(~0);
  unsigned char* current = (unsigned char*)my_string.data();
  size_t length = my_string.length();
  // process eight bytes at once
  while (length--)
    crc = (crc >> 8) ^ Crc32Lookup[(crc & 0xFF) ^ *current++];
  return crc;
}

/**
 * \brief Generate Alphabetic sequence from size_t value, A=0, Z=26, AA = 27, BA
 * = 28 = 29 T \param n index in base 26 \param array return array
 */
void findStringInv(uint64_t n, char* array)
{
  const std::uint32_t stringSizeAlphabet {alphabetSize + 1};
  const std::array<char, stringSizeAlphabet> alpha {alphabetUp};
  // If n < 27
  if (n < stringSizeAlphabet - 1) {
    array[0] = alpha[n];
    return;
  }
  // If n > 27
  std::size_t i = 0;
  while (n > 0) {
    array[i] = alpha[(--n) % alphabetSize];
    n /= alphabetSize;
    ++i;
  }
}

/**
 * \brief Fill Crc32Lookup table
 * Source: https://create.stephan-brumme.com/crc32/#slicing-by-8-overview
 */
void precompute_crc()
{
  Crc32Lookup[0] = 0;
  // compute each power of two (all numbers with exactly one bit set)
  uint32_t crc = Crc32Lookup[0x80] = Polynomial;
  for (std::uint32_t next = 0x40; next != 0; next >>= 1) {
    crc = (crc >> 1) ^ ((crc & 1) * Polynomial);
    Crc32Lookup[next] = crc;
  }

  for (std::uint32_t powerOfTwo = 2; powerOfTwo <= 0x80; powerOfTwo <<= 1) {
    uint32_t crcExtraBit = Crc32Lookup[powerOfTwo];
    for (std::uint32_t i = 1; i < powerOfTwo; i++)
      Crc32Lookup[i + powerOfTwo] = Crc32Lookup[i] ^ crcExtraBit;
  }
}