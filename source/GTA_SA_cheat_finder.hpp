#ifndef GTA_SA_HPP
#define GTA_SA_HPP

#include <algorithm>  // std::find
#include <array>  // std::array
#include <chrono>  // std::chrono
#include <cmath>  // std::ceil
#include <cstring>  // strlen
#include <iomanip>  // std::setw
#include <iostream>  // std::cout
#include <string>  // std::string
#include <tuple>  // std::pair
#include <utility>  // std::make_pair
#include <vector>  // std::vector

#include "BS_thread_pool.hpp"

#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus >= 201703L)
#  include <string_view>  // std::string_view
#  if ((defined(_MSVC_LANG) && _MSVC_LANG >= 202002L) \
       || __cplusplus >= 202002L && !defined(ANDROID) && !defined(__ANDROID__) && !defined(__EMSCRIPTEN__) \
           && !defined(__clang__))
#    include <execution>  // std::execution
#  endif
#endif

#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus >= 201703L)
#  if __has_include("omp.h")
#    include <omp.h>
#  else
#    if _MSC_VER && !__INTEL_COMPILER
#      pragma message("Can t find omp.h, please install OpenMP")
#    else
#      warning Can t find omp.h, please install OpenMP.
#    endif
#  endif
#endif

#if !defined(_OPENMP)
#  if _MSC_VER && !__INTEL_COMPILER
#    pragma message("No openMP ! Use std::thread.")
#  else
#    warning No openMP ! Use std::thread.
#  endif
#endif

#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus >= 201703L)
#  if __has_include("cuda.h")
#    ifndef BUILD_WITH_CUDA
#      define BUILD_WITH_CUDA
#    endif
#  else
#    if _MSC_VER && !__INTEL_COMPILER
#      pragma message("Can t find cuda.h, disable CUDA module")
#    else
#      warning Can t find cuda.h, disable CUDA module.
#    endif
#  endif
#endif

#if defined(BUILD_WITH_CUDA)
#  include "cuda/wrapper.hpp"
#endif

class GTA_SA
{
public:
  GTA_SA();
  void runner(const std::uint64_t&);
  void run();
  void clear();
/**
 * @brief To get JAMCRC with boost libs
 * @param my_string String input
 * @return uint32_t with JAMCRC value
 */
#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus >= 201703L)
  static auto jamcrc(std::string_view my_string, const uint32_t previousCrc32 = 0) -> std::uint32_t;
#else

#  if _MSC_VER && !__INTEL_COMPILER
#    pragma message("C++17 is not enabled, the program will be less efficient with previous standards")
#  else
#    warning C++17 is not enabled, the program will be less efficient with previous standards.
#  endif

  static auto jamcrc(const std::string& my_string, const uint32_t previousCrc32 = 0) -> std::uint32_t;
#endif

  /**
   * \brief Generate Alphabetic sequence from uint64_t value, A=1, Z=27, AA =
   * 28, AB = 29 \param n index in base 26 \param array return array
   */
  static void find_string_inv(char* array, uint64_t n);

  /**
   * \brief Source:
   * https://create.stephan-brumme.com/crc32/#slicing-by-8-overview
   */
#if defined(_MSC_VER)
  inline static std::vector<std::tuple<std::uint64_t, std::string, std::uint32_t, std::string>> results = {};
#else
  std::vector<std::tuple<std::uint64_t, std::string, std::uint32_t, std::string>> results = {};
#endif

  uint32_t max_thread_support()
  {
#if defined(_OPENMP)
    auto max_threads = static_cast<uint32_t>(omp_get_max_threads());
    if (max_threads == 0) {
      max_threads = 1;
    }
    return max_threads;
#else
    auto max_threads = static_cast<uint32_t>(std::thread::hardware_concurrency());
    if (max_threads == 0) {
      max_threads = 1;
    }
    return max_threads;
#endif
  }

  uint32_t num_thread = max_thread_support();

  // Max 1024 threads per block with CUDA 2.0 and above
  uint64_t cuda_block_size = 64;

  uint64_t min_range = 0;  // Alphabetic sequence range min
  uint64_t max_range = 0;

#if defined(BUILD_WITH_CUDA)
  uint64_t calc_mode = 2;  // 0 = std::thread, 1 = OpenMP, 2 = CUDA, 3 = OpenCL
#elif defined(_OPENMP)
  uint64_t calc_mode = 1;  // 0 = std::thread, 1 = OpenMP, 2 = CUDA, 3 = OpenCL
#else
  uint64_t calc_mode = 0;  // 0 = std::thread, 1 = OpenMP, 2 = CUDA, 3 = OpenCL
#endif

#if defined(_OPENMP)
  static constexpr bool builtWithOpenMP = true;
#else
  static constexpr bool builtWithOpenMP = false;
#endif

#if defined(BUILD_WITH_CUDA)
  static constexpr bool builtWithCUDA = true;
#else
  static constexpr bool builtWithCUDA = false;
#endif

  std::chrono::high_resolution_clock::time_point begin_time = std::chrono::high_resolution_clock::now();

  std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();

  static constexpr std::array<std::uint32_t, 87> cheat_list {
      0xDE4B237D, 0xB22A28D1, 0x5A783FAE, 0xEECCEA2B, 0x42AF1E28, 0x555FC201, 0x2A845345, 0xE1EF01EA, 0x771B83FC,
      0x5BF12848, 0x44453A17, 0xFCFF1D08, 0xB69E8532, 0x8B828076, 0xDD6ED9E9, 0xA290FD8C, 0x3484B5A7, 0x43DB914E,
      0xDBC0DD65, 0xD08A30FE, 0x37BF1B4E, 0xB5D40866, 0xE63B0D99, 0x675B8945, 0x4987D5EE, 0x2E8F84E8, 0x1A9AA3D6,
      0xE842F3BC, 0x0D5C6A4E, 0x74D4FCB1, 0xB01D13B8, 0x66516EBC, 0x4B137E45, 0x78520E33, 0x3A577325, 0xD4966D59,
      0x5FD1B49D, 0xA7613F99, 0x1792D871, 0xCBC579DF, 0x4FEDCCFF, 0x44B34866, 0x2EF877DB, 0x2781E797, 0x2BC1A045,
      0xB2AFE368, 0xFA8DD45B, 0x8DED75BD, 0x1A5526BC, 0xA48A770B, 0xB07D3B32, 0x80C1E54B, 0x5DAD0087, 0x7F80B950,
      0x6C0FA650, 0xF46F2FA4, 0x70164385, 0x885D0B50, 0x151BDCB3, 0xADFA640A, 0xE57F96CE, 0x040CF761, 0xE1B33EB9,
      0xFEDA77F7, 0x8CA870DD, 0x9A629401, 0xF53EF5A5, 0xF2AA0C1D, 0xF36345A8, 0x8990D5E1, 0xB7013B1B, 0xCAEC94EE,
      0x31F0C3CC, 0xB3B3E72A, 0xC25CDBFF, 0xD5CF4EFF, 0x680416B1, 0xCF5FDA18, 0xF01286E9, 0xA841CC0A, 0x31EA09CF,
      0xE958788A, 0x02C83A7C, 0xE49C3ED4, 0x171BA8CC, 0x86988DAE, 0x2BDD2FA1};
  /// List of cheats codes names
  const std::array<const std::string, 87> cheat_list_name {"Weapon Set 1",
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

  /**
   * \brief Source:
   * https://create.stephan-brumme.com/crc32/#slicing-by-8-overview
   */

  static constexpr std::array<uint32_t, 256> crc32_lookup = {
      0x00000000, 0x77073096, 0xEE0E612C, 0x990951BA, 0x076DC419, 0x706AF48F, 0xE963A535, 0x9E6495A3, 0x0EDB8832,
      0x79DCB8A4, 0xE0D5E91E, 0x97D2D988, 0x09B64C2B, 0x7EB17CBD, 0xE7B82D07, 0x90BF1D91, 0x1DB71064, 0x6AB020F2,
      0xF3B97148, 0x84BE41DE, 0x1ADAD47D, 0x6DDDE4EB, 0xF4D4B551, 0x83D385C7, 0x136C9856, 0x646BA8C0, 0xFD62F97A,
      0x8A65C9EC, 0x14015C4F, 0x63066CD9, 0xFA0F3D63, 0x8D080DF5, 0x3B6E20C8, 0x4C69105E, 0xD56041E4, 0xA2677172,
      0x3C03E4D1, 0x4B04D447, 0xD20D85FD, 0xA50AB56B, 0x35B5A8FA, 0x42B2986C, 0xDBBBC9D6, 0xACBCF940, 0x32D86CE3,
      0x45DF5C75, 0xDCD60DCF, 0xABD13D59, 0x26D930AC, 0x51DE003A, 0xC8D75180, 0xBFD06116, 0x21B4F4B5, 0x56B3C423,
      0xCFBA9599, 0xB8BDA50F, 0x2802B89E, 0x5F058808, 0xC60CD9B2, 0xB10BE924, 0x2F6F7C87, 0x58684C11, 0xC1611DAB,
      0xB6662D3D, 0x76DC4190, 0x01DB7106, 0x98D220BC, 0xEFD5102A, 0x71B18589, 0x06B6B51F, 0x9FBFE4A5, 0xE8B8D433,
      0x7807C9A2, 0x0F00F934, 0x9609A88E, 0xE10E9818, 0x7F6A0DBB, 0x086D3D2D, 0x91646C97, 0xE6635C01, 0x6B6B51F4,
      0x1C6C6162, 0x856530D8, 0xF262004E, 0x6C0695ED, 0x1B01A57B, 0x8208F4C1, 0xF50FC457, 0x65B0D9C6, 0x12B7E950,
      0x8BBEB8EA, 0xFCB9887C, 0x62DD1DDF, 0x15DA2D49, 0x8CD37CF3, 0xFBD44C65, 0x4DB26158, 0x3AB551CE, 0xA3BC0074,
      0xD4BB30E2, 0x4ADFA541, 0x3DD895D7, 0xA4D1C46D, 0xD3D6F4FB, 0x4369E96A, 0x346ED9FC, 0xAD678846, 0xDA60B8D0,
      0x44042D73, 0x33031DE5, 0xAA0A4C5F, 0xDD0D7CC9, 0x5005713C, 0x270241AA, 0xBE0B1010, 0xC90C2086, 0x5768B525,
      0x206F85B3, 0xB966D409, 0xCE61E49F, 0x5EDEF90E, 0x29D9C998, 0xB0D09822, 0xC7D7A8B4, 0x59B33D17, 0x2EB40D81,
      0xB7BD5C3B, 0xC0BA6CAD, 0xEDB88320, 0x9ABFB3B6, 0x03B6E20C, 0x74B1D29A, 0xEAD54739, 0x9DD277AF, 0x04DB2615,
      0x73DC1683, 0xE3630B12, 0x94643B84, 0x0D6D6A3E, 0x7A6A5AA8, 0xE40ECF0B, 0x9309FF9D, 0x0A00AE27, 0x7D079EB1,
      0xF00F9344, 0x8708A3D2, 0x1E01F268, 0x6906C2FE, 0xF762575D, 0x806567CB, 0x196C3671, 0x6E6B06E7, 0xFED41B76,
      0x89D32BE0, 0x10DA7A5A, 0x67DD4ACC, 0xF9B9DF6F, 0x8EBEEFF9, 0x17B7BE43, 0x60B08ED5, 0xD6D6A3E8, 0xA1D1937E,
      0x38D8C2C4, 0x4FDFF252, 0xD1BB67F1, 0xA6BC5767, 0x3FB506DD, 0x48B2364B, 0xD80D2BDA, 0xAF0A1B4C, 0x36034AF6,
      0x41047A60, 0xDF60EFC3, 0xA867DF55, 0x316E8EEF, 0x4669BE79, 0xCB61B38C, 0xBC66831A, 0x256FD2A0, 0x5268E236,
      0xCC0C7795, 0xBB0B4703, 0x220216B9, 0x5505262F, 0xC5BA3BBE, 0xB2BD0B28, 0x2BB45A92, 0x5CB36A04, 0xC2D7FFA7,
      0xB5D0CF31, 0x2CD99E8B, 0x5BDEAE1D, 0x9B64C2B0, 0xEC63F226, 0x756AA39C, 0x026D930A, 0x9C0906A9, 0xEB0E363F,
      0x72076785, 0x05005713, 0x95BF4A82, 0xE2B87A14, 0x7BB12BAE, 0x0CB61B38, 0x92D28E9B, 0xE5D5BE0D, 0x7CDCEFB7,
      0x0BDBDF21, 0x86D3D2D4, 0xF1D4E242, 0x68DDB3F8, 0x1FDA836E, 0x81BE16CD, 0xF6B9265B, 0x6FB077E1, 0x18B74777,
      0x88085AE6, 0xFF0F6A70, 0x66063BCA, 0x11010B5C, 0x8F659EFF, 0xF862AE69, 0x616BFFD3, 0x166CCF45, 0xA00AE278,
      0xD70DD2EE, 0x4E048354, 0x3903B3C2, 0xA7672661, 0xD06016F7, 0x4969474D, 0x3E6E77DB, 0xAED16A4A, 0xD9D65ADC,
      0x40DF0B66, 0x37D83BF0, 0xA9BCAE53, 0xDEBB9EC5, 0x47B2CF7F, 0x30B5FFE9, 0xBDBDF21C, 0xCABAC28A, 0x53B39330,
      0x24B4A3A6, 0xBAD03605, 0xCDD70693, 0x54DE5729, 0x23D967BF, 0xB3667A2E, 0xC4614AB8, 0x5D681B02, 0x2A6F2B94,
      0xB40BBE37, 0xC30C8EA1, 0x5A05DF1B, 0x2D02EF8D,
  };
};

#endif  // GTA_SA_H
