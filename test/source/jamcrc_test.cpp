#include <string>

#include "gta_sa_lib.hpp"
#include "gtest/gtest.h"

TEST(jamcrc, basic)
{
  gta::precompute_crc();

  std::string str = "AAGCEY";
  EXPECT_EQ(0x40cf761, gta::jamcrc(str));
}

