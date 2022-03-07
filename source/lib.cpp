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
  uint64_t length = my_string.length();
  // process eight bytes at once
  while (static_cast<bool>(length--)) {
    crc = (crc >> 8) ^ crc32_lookup[(crc & 0xFF) ^ *current++];
  }
  return crc;
}

/**
 * \brief Generate Alphabetic sequence from uint64_t value, A=0, Z=26, AA = 27,
 * T \param n index in base 26 \param array return array
 */
void gta::find_string_inv(char* array, uint64_t n)
{
  constexpr std::uint32_t string_size_alphabet {27};
  constexpr std::array<char, string_size_alphabet> alpha {
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"};
  // If n < 27
  if (n < 26) {
    array[0] = alpha[n];
    return;
  }
  // If n > 27
  std::uint64_t i = 0;
  while (n > 0) {
    array[i] = alpha[(--n) % 26];
    n /= 26;
    ++i;
  }
}
