#include <string>

#include "GTA_SA_cheat_finder.hpp"
#include "gtest/gtest.h"

TEST(GTA_SA, basic_calc_mode_0)
{
  GTA_SA gta_sa;
  gta_sa.min_range = 0;
  gta_sa.max_range = 0x1000;
  gta_sa.calc_mode = 0;

  gta_sa.run();

  EXPECT_EQ(gta_sa.results.size(), 0);
}

TEST(GTA_SA, basic_calc_mode_1)
{
  GTA_SA gta_sa;
  gta_sa.min_range = 0;
  gta_sa.max_range = 0x1000;
  gta_sa.calc_mode = 1;

  gta_sa.run();

  EXPECT_EQ(gta_sa.results.size(), 0);
}

TEST(GTA_SA, basic_calc_mode_2)
{
  GTA_SA gta_sa;
  gta_sa.min_range = 0;
  gta_sa.max_range = 0x1000;
  gta_sa.calc_mode = 2;

  gta_sa.run();

  EXPECT_EQ(gta_sa.results.size(), 0);
}
