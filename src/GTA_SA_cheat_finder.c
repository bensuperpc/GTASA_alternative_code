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
//  Modified: 30, July, 2021                                //
//  file: crypto.c                                          //
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

#include "GTA_SA_cheat_finder.h"
#include <stddef.h>

bool search_for_number(const unsigned int *a, unsigned int search)
{
  const size_t array_size = sizeof(*a)/sizeof(a[0]);
  for(size_t i = 0; i < array_size; i++)
  {
      if(a[i] == search)      // or if(*(a + i) == search)
      {
          return true;
      }
  }
  return false;
}


unsigned int jamcrc(const unsigned char* my_string) {
  uint32_t crc = ~0;
  unsigned char *current = (unsigned char *)my_string;
  size_t length = sizeof(*my_string)/sizeof(my_string[0]);
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
 * \brief Generate Alphabetic sequence from size_t value, A=0, Z=26, AA = 27, BA = 28
 * = 29 \tparam size_t \param n index in base 26 \param array return array
 */
void findStringInv(size_t n, char *array) {
  const unsigned int stringSizeAlphabet = sizeof(alphabetSize)/sizeof(alphabetSize[0]) + 1;
  const char alpha[stringSizeAlphabet] = {alphabetUp};
  // If n < 27
  if (n < stringSizeAlphabet - 1) {
    array[0] = alpha[n];
  }
  // If n > 27
  size_t i = 0;
  while (n > 0) {
    array[i] = alpha[(--n) % (size_t)alphabetSize];
    n /= (size_t)alphabetSize;
    ++i;
  }
}

int main(int arc, char *argv[]) {
  size_t from_range = 0; // Alphabetic sequence range min
  if (arc >= 3) {
    from_range = atoi(argv[1]);
  }
  // if you want begin on higer range, 1 = A
  // 141167095653376 = ~17 days on I7 9750H
  // 5429503678976 = ~14h on I7 9750H
  // 208827064576 = ~28 min on I7 9750H
  // 8031810176 = ~1 min on I7 9750H
  // 1544578880 = ~11 sec on I7 9750H
  // 308915776 = 2 sec on I7 9750H

  size_t to_range =
      600000000; // Alphabetic sequence range max, must be > from_range !
  if (arc == 2) {
    to_range = atoi(argv[1]);
  }
  if (arc >= 3) {
    to_range = atoi(argv[2]);
  }
  assert(from_range <
         to_range);       // Test if begining value is highter than end value
  //assert(from_range > 0); // Test forbiden value
  printf("Number of calculations:");
  printf("%zu\n", (to_range - from_range));
  printf("\n");

  // Display Alphabetic sequence range
  char tmp1[29] = {0};
  char tmp2[29] = {0};
  findStringInv(from_range, tmp1);
  findStringInv(to_range, tmp2);

  printf("From: ");
  printf("%s", tmp1);
  printf(" to: ");
  printf("%s", tmp2);
  printf(" Alphabetic sequence\n");
  printf("\n");

  char tmp[29] = {0}; // Temp array
  uint32_t crc = 0;   // CRC value
  auto &&t1 = Clock::now();
#if defined(_OPENMP)
#pragma omp parallel for schedule(auto) shared(results) firstprivate(tmp, crc)
#endif
  for (size_t i = from_range; i < to_range; i++) {
    findStringInv(i, tmp); // Generate Alphabetic sequence from size_t
                                   // value, A=1, Z=27, AA = 28, AB = 29
    crc = jamcrc(tmp);             // JAMCRC

    if(search_for_number(cheat_list, crc) == true) 
    {

    }

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
            << ((double)(to_range - from_range) /
                std::chrono::duration_cast<std::chrono::duration<double>>(t2 -
                                                                          t1)
                    .count()) /
                   1000000
            << " MOps/sec" << std::endl;

  return EXIT_SUCCESS;
}
/** @} */ // end of group2
