//////////////////////////////////////////////////////////////
//   ____                                                   //
//  | __ )  ___ _ __  ___ _   _ _ __   ___ _ __ _ __   ___  //
//  |  _ \ / _ \ '_ \/ __| | | | '_ \ / _ \ '__| '_ \ / __| //
//  | |_) |  __/ | | \__ \ |_| | |_) |  __/ |  | |_) | (__  //
//  |____/ \___|_| |_|___/\__,_| .__/ \___|_|  | .__/ \___| //
//                             |_|             |_|          //
//////////////////////////////////////////////////////////////
//                                                          //
//  BenLib, 2021                                            //
//  Created: 26, February, 2021                             //
//  Modified: 14, June, 2021                                //
//  file: crypto.cpp                                        //
//  Crypto                                                  //
//  Source:
//          http://stackoverflow.com/questions/8710719/generating-an-alphabetic-sequence-in-java
//          https://github.com/theo546/stuff //
//          https://stackoverflow.com/a/19299611/10152334 //
//          https://gms.tf/stdfind-and-memchr-optimizations.html //
//          https://medium.com/applied/applied-c-align-array-elements-32af40a768ee
//          https://create.stephan-brumme.com/crc32/ //
//          https://rosettacode.org/wiki/Generate_lower_case_ASCII_alphabet //
//          https://web.archive.org/web/20090204140550/http://www.maxbot.com/gta/3wordcheatsdumpsorted.txt
//          https://www.codeproject.com/Articles/663443/Cplusplus-is-Fun-Optimal-Alphabetical-Order
//          https://cppsecrets.com/users/960210997110103971089710711510497116484964103109971051084699111109/Given-integer-n-find-the-nth-string-in-this-sequence-A-B-C-Z-AA-AB-AC-ZZ-AAA-AAB-AAZ-ABA-.php
//          https://www.careercup.com/question?id=14276663 //
//          https://stackoverflow.com/a/55074804/10152334 //
//          https://web.archive.org/web/20090204140550/http://www.maxbot.com/gta/3wordcheatsdumpsorted.txt
//          https://stackoverflow.com/questions/26429360/crc32-vs-crc32c //
//          https://create.stephan-brumme.com/crc32/#half-byte
//  OS: ALL                                                 //
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

#include <algorithm> // std::find
#include <chrono>
#include <cstring> // strlen
#include <cassert>
#include <iomanip>  // std::setw
#include <iostream> // std::cout
#include <string>   // std::string
#include <string_view> // std::string_view
#include <tuple>
#include <utility> // std::make_pair
#include <vector> // std::vector
typedef std::chrono::high_resolution_clock Clock;

#if __has_include("omp.h")
#include <omp.h>
#endif

#if !defined(_OPENMP)
#warning No openMP ! Only use 1 thread.
#endif

/** @brief If you want display less informations, comment it */
#define MORE_INFO

/** @brief For debug mode */
#define DNDEBUG

/** @brief Define alphabetic seq with upercase */
#define alphabetUp "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

/** @brief Size of alphabet */
constexpr std::uint32_t alphabetSize{26};

/** @brief List of CRC32/JAMCRC hash of cheats codes */
const std::array<unsigned int, 87> cheat_list{
    0xDE4B237D, 0xB22A28D1, 0x5A783FAE, 0xEECCEA2B, 0x42AF1E28, 0x555FC201,
    0x2A845345, 0xE1EF01EA, 0x771B83FC, 0x5BF12848, 0x44453A17, 0xFCFF1D08,
    0xB69E8532, 0x8B828076, 0xDD6ED9E9, 0xA290FD8C, 0x3484B5A7, 0x43DB914E,
    0xDBC0DD65, 0xD08A30FE, 0x37BF1B4E, 0xB5D40866, 0xE63B0D99, 0x675B8945,
    0x4987D5EE, 0x2E8F84E8, 0x1A9AA3D6, 0xE842F3BC, 0x0D5C6A4E, 0x74D4FCB1,
    0xB01D13B8, 0x66516EBC, 0x4B137E45, 0x78520E33, 0x3A577325, 0xD4966D59,
    0x5FD1B49D, 0xA7613F99, 0x1792D871, 0xCBC579DF, 0x4FEDCCFF, 0x44B34866,
    0x2EF877DB, 0x2781E797, 0x2BC1A045, 0xB2AFE368, 0xFA8DD45B, 0x8DED75BD,
    0x1A5526BC, 0xA48A770B, 0xB07D3B32, 0x80C1E54B, 0x5DAD0087, 0x7F80B950,
    0x6C0FA650, 0xF46F2FA4, 0x70164385, 0x885D0B50, 0x151BDCB3, 0xADFA640A,
    0xE57F96CE, 0x040CF761, 0xE1B33EB9, 0xFEDA77F7, 0x8CA870DD, 0x9A629401,
    0xF53EF5A5, 0xF2AA0C1D, 0xF36345A8, 0x8990D5E1, 0xB7013B1B, 0xCAEC94EE,
    0x31F0C3CC, 0xB3B3E72A, 0xC25CDBFF, 0xD5CF4EFF, 0x680416B1, 0xCF5FDA18,
    0xF01286E9, 0xA841CC0A, 0x31EA09CF, 0xE958788A, 0x02C83A7C, 0xE49C3ED4,
    0x171BA8CC, 0x86988DAE, 0x2BDD2FA1};

#ifdef MORE_INFO
/// List of cheats codes names
const std::array<const std::string, 87> cheat_list_name{
    "Weapon Set 1",
    "Weapon Set 2",
    "Weapon Set 3",
    "Health, Armor, $250k, Repairs car",
    "Increase Wanted Level +2",
    "Clear Wanted Level",
    "Sunny Weather",
    "Very Sunny Weather",
    "Overcast Weather",
    "Rainy Weather",
    "Foggy Weather",
    "Faster Clock",
    "N°12",
    "N°13",
    "People attack each other with golf clubs",
    "Have a bounty on your head",
    "Everyone is armed",
    "Spawn Rhino",
    "Spawn Bloodring Banger",
    "Spawn Rancher",
    "Spawn Racecar",
    "Spawn Racecar",
    "Spawn Romero",
    "Spawn Stretch",
    "Spawn Trashmaster",
    "Spawn Caddy",
    "Blow Up All Cars",
    "Invisible car",
    "All green lights",
    "Aggressive Drivers",
    "Pink CArs",
    "Black Cars",
    "Fat Body",
    "Muscular Body",
    "Skinny Body",
    "People attack with Rocket Launchers",
    "N°41",
    "N°42",
    "Gangs Control the Streets",
    "N°44",
    "Slut Magnet",
    "N°46",
    "N°47",
    "Cars Fly",
    "N°49",
    "N°50",
    "Spawn Vortex Hovercraft",
    "Smash n' Boom",
    "N°53",
    "N°54",
    "N°55",
    "Orange Sky",
    "Thunderstorm",
    "Sandstorm",
    "N°59",
    "N°60",
    "Infinite Health",
    "Infinite Oxygen",
    "Have Parachute",
    "N°64",
    "Never Wanted",
    "N°66",
    "Mega Punch",
    "Never Get Hungry",
    "N°69",
    "N°70",
    "N°71",
    "N°72",
    "Full Weapon Aiming While Driving",
    "N°74",
    "Traffic is Country Vehicles",
    "Recruit Anyone (9mm)",
    "Get Born 2 Truck Outfit",
    "N°78",
    "N°79",
    "N°80",
    "L3 Bunny Hop",
    "N°82",
    "N°83",
    "N°84",
    "Spawn Quad",
    "Spawn Tanker Truck",
    "Spawn Dozer",
    "pawn Stunt Plane",
    "Spawn Monster"};
#endif

std::vector<std::tuple<std::size_t, std::string, unsigned int>> results =
    {}; // Stock results after calculations

/**
 * @brief To get JAMCRC with boost libs
 * @param my_string String input
 * @return uint32_t with JAMCRC value
 */
unsigned int jamcrc(const std::string_view my_string);
unsigned int jamcrc(const std::string_view my_string) {
  uint32_t crc = ~0;
  unsigned char *current = (unsigned char *)my_string.data();
  size_t length = my_string.length();
  static uint32_t lut[16] = {0x00000000, 0x1DB71064, 0x3B6E20C8, 0x26D930AC,
                             0x76DC4190, 0x6B6B51F4, 0x4DB26158, 0x5005713C,
                             0xEDB88320, 0xF00F9344, 0xD6D6A3E8, 0xCB61B38C,
                             0x9B64C2B0, 0x86D3D2D4, 0xA00AE278, 0xBDBDF21C};
  while (length--) {
    crc = lut[(crc ^ *current) & 0x0F] ^ (crc >> 4);
    crc = lut[(crc ^ (*current >> 4)) & 0x0F] ^ (crc >> 4);
    current++;
  }
  return crc;
}

/**
 * \brief Generate Alphabetic sequence from size_t value, A=1, Z=27, AA = 28, AB
 * = 29 \tparam T \param n index in base 26 \param array return array
 */
template <class T> void findStringInv(T n, char *array);
template <class T> void findStringInv(T n, char *array) {
  constexpr std::uint32_t stringSizeAlphabet{alphabetSize + 1};
  constexpr std::array<char, stringSizeAlphabet> alpha{alphabetUp};
  // If n < 27
  if (n < stringSizeAlphabet) {
    array[0] = alpha[n - 1];
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

int main(int arc, char *argv[]) {
  std::ios_base::sync_with_stdio(false);

  const size_t from_range = 1; // Alphabetic sequence range min, change it only
                               // if you want begin on higer range, 1 = A
  // 141167095653376 = ~17 days on I7 9750H
  // 5429503678976 = ~14h on I7 9750H
  // 208827064576 = ~28 min on I7 9750H
  // 8031810176 = ~1 min on I7 9750H
  // 1544578880 = ~11 sec on I7 9750H
  // 308915776 = 2 sec on I7 9750H
  const size_t to_range =
      308915776; // Alphabetic sequence range max, must be > from_range !

#ifdef DNDEBUG
  assert(from_range <
         to_range);       // Test if begining value is highter than end value
  assert(from_range > 0); // Test forbiden value
#endif

#ifdef MORE_INFO
  std::cout << "Number of calculations: " << (to_range - from_range)
            << std::endl;
  std::cout << "" << std::endl;
  // Display Alphabetic sequence range
  char tmp1[29] = {0};
  char tmp2[29] = {0};
  findStringInv<size_t>(from_range, tmp1);
  findStringInv<size_t>(to_range, tmp2);
  std::cout << "From: " << tmp1 << " to: " << tmp2 << " Alphabetic sequence"
            << std::endl;
  std::cout << "" << std::endl;
#endif

  char tmp[29] = {0}; // Temp array
  uint32_t crc = 0;   // CRC value
  auto &&t1 = Clock::now();
#if defined(_OPENMP)
#pragma omp parallel for schedule(auto) shared(results) firstprivate(tmp, crc)
#endif
  for (std::size_t i = from_range; i < to_range; i++) {
    findStringInv<size_t>(i, tmp); // Generate Alphabetic sequence from size_t
                                   // value, A=1, Z=27, AA = 28, AB = 29
    crc = jamcrc(
        tmp); // JAMCRC
    if (std::find(std::begin(cheat_list), std::end(cheat_list), crc) !=
        std::end(cheat_list)) {             // If crc is present in Array
      std::reverse(tmp, tmp + strlen(tmp)); // Invert char array
      results.emplace_back(std::make_tuple(
          i, std::string(tmp),
          crc)); // Save result: calculation position, Alphabetic sequence, CRC
    }
  }
  auto &&t2 = Clock::now();
  sort(results.begin(), results.end());
  for (auto &&result : results) {
    std::cout << std::left << std::setw(14) << std::dec << std::get<0>(result)
              << std::left << std::setw(12) << std::get<1>(result) << "0x"
              << std::hex << std::left << std::setw(12) << std::get<2>(result);
    std::cout << std::endl;
  }
  std::cout << "Time: "
            << std::chrono::duration_cast<std::chrono::duration<double>>(t2 -
                                                                         t1)
                   .count()
            << " sec" << std::endl;
  std::cout << "This program execute: " << std::fixed
            << ((to_range - from_range) /
                std::chrono::duration_cast<std::chrono::duration<double>>(t2 -
                                                                          t1)
                    .count()) /
                   1000000
            << " MOps/sec" << std::endl;

  return EXIT_SUCCESS;
}
/** @} */ // end of group2