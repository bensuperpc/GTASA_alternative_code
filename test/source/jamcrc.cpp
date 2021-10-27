#include <string>

#include "gta_sa_lib.hpp"

auto main() -> int
{
  std::string str = "BARBAPAPA";
  auto jamcrc = gta::jamcrc(str);

  return 0;
}
