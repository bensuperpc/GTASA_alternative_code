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

#include "GTA_SA_cheat_finder.hpp"

/**
 * \brief Source: https://create.stephan-brumme.com/crc32/#slicing-by-8-overview
 */
#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus >= 201703L)
unsigned int jamcrc(std::string_view my_string) {
#else
#warning C++17 is not enabled, the program will be less efficient with previous standards.
unsigned int jamcrc(const std::string my_string) {
#endif
  uint32_t crc = static_cast<uint32_t>(~0);
  unsigned char *current = (unsigned char *)my_string.data();
  size_t length = my_string.length();
  // process eight bytes at once
  while (length >= 8) {
    uint32_t one = *current++ ^ crc;
    uint32_t two = *current++;
    crc = Crc32Lookup[7][one & 0xFF] ^ Crc32Lookup[6][(one >> 8) & 0xFF] ^
          Crc32Lookup[5][(one >> 16) & 0xFF] ^ Crc32Lookup[4][one >> 24] ^
          Crc32Lookup[3][two & 0xFF] ^ Crc32Lookup[2][(two >> 8) & 0xFF] ^
          Crc32Lookup[1][(two >> 16) & 0xFF] ^ Crc32Lookup[0][two >> 24];
    length -= 8;
  }
  unsigned char *currentChar = (unsigned char *)current;
  // remaining 1 to 7 bytes
  while (length--)
    crc = (crc >> 8) ^ Crc32Lookup[0][(crc & 0xFF) ^ *currentChar++];
  return crc;
}

/**
 * \brief Generate Alphabetic sequence from size_t value, A=0, Z=26, AA = 27, BA
 * = 28 = 29 \tparam T \param n index in base 26 \param array return array
 */
template <class T> void findStringInv(T n, char *array) {
  constexpr std::uint32_t stringSizeAlphabet{alphabetSize + 1};
  constexpr std::array<char, stringSizeAlphabet> alpha{alphabetUp};
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

int main(int arc, char *argv[]) {
  std::ios_base::sync_with_stdio(false); // Improve std::cout speed

  // same as before
  for (unsigned int i = 0; i <= 0xFF; i++) {
    uint32_t crc = i;
    for (unsigned int j = 0; j < 8; j++)
      crc = (crc >> 1) ^ ((crc & 1) * Polynomial);
    Crc32Lookup[0][i] = crc;
  }
  for (unsigned int i = 0; i <= 0xFF; i++) {
    // for Slicing-by-4 and Slicing-by-8
    Crc32Lookup[1][i] =
        (Crc32Lookup[0][i] >> 8) ^ Crc32Lookup[0][Crc32Lookup[0][i] & 0xFF];
    Crc32Lookup[2][i] =
        (Crc32Lookup[1][i] >> 8) ^ Crc32Lookup[0][Crc32Lookup[1][i] & 0xFF];
    Crc32Lookup[3][i] =
        (Crc32Lookup[2][i] >> 8) ^ Crc32Lookup[0][Crc32Lookup[2][i] & 0xFF];
    // only Slicing-by-8
    Crc32Lookup[4][i] =
        (Crc32Lookup[3][i] >> 8) ^ Crc32Lookup[0][Crc32Lookup[3][i] & 0xFF];
    Crc32Lookup[5][i] =
        (Crc32Lookup[4][i] >> 8) ^ Crc32Lookup[0][Crc32Lookup[4][i] & 0xFF];
    Crc32Lookup[6][i] =
        (Crc32Lookup[5][i] >> 8) ^ Crc32Lookup[0][Crc32Lookup[5][i] & 0xFF];
    Crc32Lookup[7][i] =
        (Crc32Lookup[6][i] >> 8) ^ Crc32Lookup[0][Crc32Lookup[6][i] & 0xFF];
  }

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
  assert(min_range <
         max_range); // Test if begining value is highter than end value
  // assert(min_range > 0); // Test forbiden value

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
  for (std::size_t i = min_range; i < max_range; i++) {
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
/** @} */ // end of group2
