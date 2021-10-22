#include <algorithm>  // std::find
#include <array>  // std::array
#include <chrono>  // std::chrono
#include <cstring>  // strlen
#include <iomanip>  // std::setw
#include <iostream>  // std::cout
#include <string>  // std::string
#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus >= 201703L)
#  include <string_view>  // std::string_view
#  if ((defined(_MSVC_LANG) && _MSVC_LANG >= 202002L) \
       || __cplusplus >= 202002L && !defined(ANDROID) && !defined(__ANDROID__) \
           && !defined(__EMSCRIPTEN__) && !defined(__clang__))
#    include <execution>  // std::execution
#  endif
#endif
#include <tuple>  // std::pair
#include <utility>  // std::make_pair
#include <vector>  // std::vector

#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus >= 201703L)
#  if __has_include("omp.h")
#    include <omp.h>
#  endif
#endif

#if !defined(_OPENMP)
#  warning No openMP ! Only use 1 thread.
#endif

/** @brief Define alphabetic seq with upercase */
#define alphabetUp "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

namespace gta
{

/** @brief Size of alphabet */
static const std::uint32_t alphabet_size {26};

/** @brief List of CRC32/JAMCRC hash of cheats codes */
static const std::array<std::uint32_t, 87> cheat_list {
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
/*
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
*/

/**
 * @brief To get JAMCRC with boost libs
 * @param my_string String input
 * @return uint32_t with JAMCRC value
 */
#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus >= 201703L)
std::uint32_t jamcrc(std::string_view my_string);
#else
#warning C++17 is not enabled, the program will be less efficient with previous standards.
auto jamcrc(const std::string& my_string) -> std::uint32_t;
#endif

/**
 * \brief Generate Alphabetic sequence from size_t value, A=1, Z=27, AA = 28, AB
 * = 29 \param n index in base 26 \param array return array
 */
void findStringInv(uint64_t n, char* array);

static const uint32_t Polynomial = 0xEDB88320;

/**
 * \brief Source: https://create.stephan-brumme.com/crc32/#slicing-by-8-overview
 */
void precompute_crc();

static std::array<uint32_t, 256> crc32_lookup = {0};

} // namespace gta