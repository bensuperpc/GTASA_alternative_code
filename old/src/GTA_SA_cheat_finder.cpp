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
//  Modified: 14, October, 2021                             //
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

#include "GTA_SA_cheat_finder.hpp"
#include <cstdlib>

/**
 * \brief Source: https://create.stephan-brumme.com/crc32/#slicing-by-8-overview
 */
#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus >= 201703L)
std::uint32_t jamcrc(std::string_view my_string) {
#else
#warning C++17 is not enabled, the program will be less efficient with previous standards.
std::uint32_t jamcrc(const std::string &my_string) {
#endif
  uint32_t crc = static_cast<uint32_t>(~0);
  unsigned char *current = (unsigned char *)my_string.data();
  size_t length = my_string.length();
  // process eight bytes at once
  while (length--)
    crc = (crc >> 8) ^ Crc32Lookup[(crc & 0xFF) ^ *current++];
  return crc;
}

/**
 * \brief Generate Alphabetic sequence from size_t value, A=0, Z=26, AA = 27, BA
 * = 28 = 29 \tparam T \param n index in base 26 \param array return array
 */
template <class T> void findStringInv(T n, char *array) {
  const std::uint32_t stringSizeAlphabet{alphabetSize + 1};
  const std::array<char, stringSizeAlphabet> alpha{alphabetUp};
  // If n < 27
  if (n < stringSizeAlphabet - 1) {
    array[0] = alpha[n];
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

/**
 * \brief Fill Crc32Lookup table
 * Source: https://create.stephan-brumme.com/crc32/#slicing-by-8-overview
 */
void precompute_crc() {
  Crc32Lookup[0] = 0;
  // compute each power of two (all numbers with exactly one bit set)
  uint32_t crc = Crc32Lookup[0x80] = Polynomial;
  for (std::uint32_t next = 0x40; next != 0; next >>= 1) {
    crc = (crc >> 1) ^ ((crc & 1) * Polynomial);
    Crc32Lookup[next] = crc;
  }

  for (std::uint32_t powerOfTwo = 2; powerOfTwo <= 0x80; powerOfTwo <<= 1) {
    uint32_t crcExtraBit = Crc32Lookup[powerOfTwo];
    for (std::uint32_t i = 1; i < powerOfTwo; i++)
      Crc32Lookup[i + powerOfTwo] = Crc32Lookup[i] ^ crcExtraBit;
  }
}

int main(int arc, char *argv[]) {
  std::ios_base::sync_with_stdio(false); // Improve std::cout speed

  precompute_crc(); // Fill Crc32Lookup table

  size_t min_range = 0; // Alphabetic sequence range min
  if (arc >= 3) {
    min_range = static_cast<size_t>(std::stoll(argv[1]));
  }

  size_t max_range =
      600000000; // Alphabetic sequence range max, must be > min_range !
  if (arc == 2) {
    max_range = static_cast<size_t>(std::stoll(argv[1]));
  }
  if (arc >= 3) {
    max_range = static_cast<size_t>(std::stoll(argv[2]));
  }

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
  char tmp1[29] = {0};
  char tmp2[29] = {0};
  findStringInv<size_t>(min_range, tmp1);
  findStringInv<size_t>(max_range, tmp2);
  std::cout << "From: " << tmp1 << " to: " << tmp2 << " Alphabetic sequence"
            << std::endl;
  std::cout << "" << std::endl;

  char tmp[29] = {0}; // Temp array
  uint32_t crc = 0;   // CRC value
  auto &&t1 = Clock::now();
#if defined(_OPENMP)
#pragma omp parallel for schedule(auto) shared(results) firstprivate(tmp, crc)
#endif
  for (std::size_t i = min_range; i <= max_range; i++) {
    findStringInv<size_t>(i, tmp); // Generate Alphabetic sequence from size_t
                                   // value, A=1, Z=27, AA = 28, AB = 29
    crc = jamcrc(tmp);             // JAMCRC
#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 202002L) ||                         \
     __cplusplus >= 202002L && !defined(ANDROID) && !defined(__ANDROID__) &&   \
         !defined(__EMSCRIPTEN__) && !defined(__clang__))
    if (std::find(std::execution::unseq, std::begin(cheat_list),
                  std::end(cheat_list), crc) != std::end(cheat_list)) {
#else
    if (std::find(std::begin(cheat_list), std::end(cheat_list), crc) !=
        std::end(cheat_list)) {
#endif                                      // If crc is present in Array
      std::reverse(tmp, tmp + strlen(tmp)); // Invert char array
      results.emplace_back(std::make_tuple(
          i, std::string(tmp),
          crc)); // Save result: calculation position, Alphabetic sequence, CRC
    }
  }
  auto &&t2 = Clock::now();

  sort(results.begin(), results.end()); // Sort results

  std::cout << std::left << std::setw(18) << "Iter. N°" << std::left
            << std::setw(15) << "Code" << std::left << std::setw(15)
            << "JAMCRC value" << std::endl;

  for (auto &&result : results) {
    std::cout << std::left << std::setw(17) << std::dec << std::get<0>(result)
              << std::left << std::setw(15) << std::get<1>(result) << "0x"
              << std::hex << std::left << std::setw(15) << std::get<2>(result)
              << std::endl;
  }

  std::cout << "Time: "
            << std::chrono::duration_cast<std::chrono::duration<double>>(t2 -
                                                                         t1)
                   .count()
            << " sec" << std::endl; // Display time
  std::cout << "This program execute: " << std::fixed
            << ((double)(max_range - min_range) /
                std::chrono::duration_cast<std::chrono::duration<double>>(t2 -
                                                                          t1)
                    .count()) /
                   1000000
            << " MOps/sec" << std::endl; // Display perf

  return EXIT_SUCCESS;
}