#include <string>

#include "GTA_SA_cheat_finder.hpp"
#include "gtest/gtest.h"

TEST(jamcrc, basic1)
{
  std::string str = "";
  std::reverse(str.begin(), str.end());

  EXPECT_EQ("", str);

  EXPECT_NE(0x0, GTA_SA::jamcrc(str));
  EXPECT_EQ(0xffffffff, GTA_SA::jamcrc(str));
}

TEST(jamcrc, basic2)
{
  std::string str = "ASNAEB";
  std::reverse(str.begin(), str.end());

  EXPECT_EQ("BEANSA", str);

  EXPECT_NE(0x0, GTA_SA::jamcrc(str));
  EXPECT_EQ(0x555fc201, GTA_SA::jamcrc(str));
}

TEST(jamcrc, basic3)
{
  std::string str = "ASBHGRB";
  std::reverse(str.begin(), str.end());

  EXPECT_EQ("BRGHBSA", str);

  EXPECT_NE(0x0, GTA_SA::jamcrc(str));
  EXPECT_EQ(0xa7613f99, GTA_SA::jamcrc(str));
}

TEST(jamcrc, basic4)
{
  std::string str = "XICWMD";
  std::reverse(str.begin(), str.end());

  EXPECT_EQ("DMWCIX", str);

  EXPECT_NE(0x0, GTA_SA::jamcrc(str));
  EXPECT_EQ(0x1a9aa3d6, GTA_SA::jamcrc(str));
}

TEST(jamcrc, basic5)
{
  std::string str = "LGBTQIA+";
  std::reverse(str.begin(), str.end());

  EXPECT_EQ("+AIQTBGL", str);

  EXPECT_NE(0x0, GTA_SA::jamcrc(str));
  EXPECT_EQ(0x6ba88a6, GTA_SA::jamcrc(str));
}

TEST(jamcrc, basic6)
{
  std::string str = "intergouvernementalisations";
  std::reverse(str.begin(), str.end());

  EXPECT_EQ("snoitasilatnemenrevuogretni", str);

  EXPECT_NE(0x0, GTA_SA::jamcrc(str));
  EXPECT_EQ(0x1a384955, GTA_SA::jamcrc(str));
}
