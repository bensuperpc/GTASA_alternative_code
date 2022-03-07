#include <catch2/catch.hpp>

#include "lib.hpp"

TEST_CASE("Name is GTA_SA_cheat_finder", "[library]")
{
  library lib;
  REQUIRE(lib.name == "GTA_SA_cheat_finder");
}
