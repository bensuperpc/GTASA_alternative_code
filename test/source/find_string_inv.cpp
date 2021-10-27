#include <array>

#include "gta_sa_lib.hpp"

auto main() -> int
{
  const auto array_size = 29;

  std::array<char, array_size> tmp1 = {0};
  gta::find_string_inv(tmp1.data(), 0);

  std::array<char, array_size> tmp2 = {0};
  gta::find_string_inv(tmp2.data(), 255);

  return 0;
}
