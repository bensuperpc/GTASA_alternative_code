#include <chrono>  // std::chrono
#include <iostream>  // std::cout
#include <sstream>  // std::istringstream

#include "gta_sa_lib.hpp"

#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus >= 201703L)
#  if __has_include("omp.h")
#    include <omp.h>
#  endif
#endif

#if !defined(_OPENMP)
#  if _MSC_VER && !__INTEL_COMPILER
#    pragma message("No openMP ! Only use 1 thread.")
#  else
#    warning No openMP ! Only use 1 thread.
#  endif
#endif

using cpp_clock = std::chrono::high_resolution_clock;

auto main(int argc, char* argv[]) -> int
{
  std::ios_base::sync_with_stdio(false);  // Improve std::cout speed
  std::cout.setf(std::ios::left);

  std::vector<std::tuple<std::uint64_t, std::string, std::uint32_t>> results =
      {};  // Stock results after calculations

  uint64_t min_range = 0;  // Alphabetic sequence range min
  uint64_t max_range =
      0;  // Alphabetic sequence range max, must be > min_range !

  std::vector<std::string> args(argv + 1, argv + argc);
  std::string infname;
  std::string outfname;

  // args[i] instead of *i -- don't tell anyone! ;)
  for (auto i = args.begin(); i != args.end(); ++i) {
    if (*i == "-h" || *i == "--help") {
      std::cout << "Syntax: GTA_SA_cheat_finder --min <from (uint64_t)> --max "
                   "<to (uint64_t)>"
                << std::endl;
      return EXIT_SUCCESS;
    }
    if (*i == "--min") {
      std::istringstream iss(*++i);
      if (!(iss >> min_range)) {
        std::cout << "Error, non-numeric character !" << std::endl;
        return EXIT_FAILURE;
      }
    } else if (*i == "--max") {
      std::istringstream iss(*++i);
      if (!(iss >> max_range)) {
        std::cout << "Error, non-numeric character !" << std::endl;
        return EXIT_FAILURE;
      }
    }
  }

  args.clear();  // Remove all element
  args.shrink_to_fit();  // Free memory

  if (min_range > max_range) {
    std::cout << "Min range value: '" << min_range
              << "' can't be greater than Max range value: '" << max_range
              << "'" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Number of calculations: " << (max_range - min_range)
            << std::endl;
  std::cout << "" << std::endl;
  // Display Alphabetic sequence range
  std::array<char, 29> tmp1 = {0};
  std::array<char, 29> tmp2 = {0};

  gta::find_string_inv(tmp1.data(), min_range);
  gta::find_string_inv(tmp2.data(), max_range);
  std::cout << "From: " << tmp1.data() << " to: " << tmp2.data()
            << " Alphabetic sequence" << std::endl;
  std::cout << "" << std::endl;

  std::array<char, 29> tmp = {0};  // Temp array
  uint32_t crc = 0;  // CRC value
  auto&& t1 = cpp_clock::now();
#if defined(_OPENMP)
#  pragma omp parallel for schedule(auto) shared(results) firstprivate(tmp, crc)
#endif
  for (std::uint64_t i = min_range; i <= max_range; i++) {
    gta::find_string_inv(tmp.data(),
                         i);  // Generate Alphabetic sequence from uint64_t
                              // value, A=1, Z=27, AA = 28, AB = 29
    crc = gta::jamcrc(tmp.data());  // JAMCRC
#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 202002L) \
     || __cplusplus >= 202002L && !defined(ANDROID) && !defined(__ANDROID__) \
         && !defined(__EMSCRIPTEN__) && !defined(__clang__))
    if (std::find(std::execution::unseq,
                  std::begin(gta::cheat_list),
                  std::end(gta::cheat_list),
                  crc)
        != std::end(gta::cheat_list))
    {
#else
    if (std::find(std::begin(gta::cheat_list), std::end(gta::cheat_list), crc)
        != std::end(gta::cheat_list))
    {
#endif
      // If crc is present in Array
      std::reverse(tmp.data(),
                   tmp.data() + strlen(tmp.data()));  // Invert char array
      results.emplace_back(std::make_tuple(
          i,
          std::string(tmp.data()),
          crc));  // Save result: calculation position, Alphabetic sequence, CRC
    }
  }
  auto&& t2 = cpp_clock::now();

  sort(results.begin(), results.end());  // Sort results

  constexpr auto display_val = 15;

  std::cout << std::setw(display_val + 4) << "Iter. N°"
            << std::setw(display_val) << "Code" << std::setw(display_val)
            << "JAMCRC value" << std::endl;

  for (auto& result : results) {
    std::cout << std::setw(display_val + 3) << std::dec << std::get<0>(result)
              << std::setw(display_val) << std::get<1>(result) << "0x"
              << std::hex << std::setw(display_val) << std::get<2>(result)
              << std::endl;
  }
  std::cout << "Time: "
            << std::chrono::duration_cast<std::chrono::duration<double>>(t2
                                                                         - t1)
                   .count()
            << " sec" << std::endl;  // Display time
  std::cout << "This program execute: " << std::fixed
            << (static_cast<double>(max_range - min_range)
                / std::chrono::duration_cast<std::chrono::duration<double>>(
                      t2 - t1)
                      .count())
          / 1000000
            << " MOps/sec" << std::endl;  // Display perf

  return EXIT_SUCCESS;
}