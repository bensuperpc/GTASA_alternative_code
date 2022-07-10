#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "GTA_SA_cheat_finder.hpp"

auto main(int argc, char* argv[]) -> int
{
  GTA_SA gta_sa;

  // std::ios_base::sync_with_stdio(false);  // Improve std::cout speed

  std::vector<std::string> args(argv + 1, argv + argc);

  for (auto i = args.begin(); i != args.end(); ++i) {
    if (*i == "-h" || *i == "--help") {
      std::cout << "Syntax: GTA_SA_cheat_finder --min <from (uint64_t)> --max "
                   "<to (uint64_t)>"
                   "--calc-mode <0-2> 0: std::thread, 1: OpenMP, 2: CUDA>"
                << std::endl;
      return EXIT_SUCCESS;
    }
    if (*i == "--min") {
      std::istringstream iss(*++i);
      if (!(iss >> gta_sa.min_range)) {
        std::cout << "Error, non-numeric character !" << std::endl;
        return EXIT_FAILURE;
      }
    } else if (*i == "--max") {
      std::istringstream iss(*++i);
      if (!(iss >> gta_sa.max_range)) {
        std::cout << "Error, non-numeric character !" << std::endl;
        return EXIT_FAILURE;
      }
    } else if (*i == "--calc-mode") {
      std::istringstream iss(*++i);
      if (!(iss >> gta_sa.calc_mode)) {
        std::cout << "Error, non-numeric character !" << std::endl;
        return EXIT_FAILURE;
      }
    }
  }

  // gta_sa.num_thread
  // gta_sa.min_range
  // gta_sa.max_range
  // gta_sa.num_thread
  // gta_sa.cuda_block_size
  // gta_sa.calc_mode

  // Launch operation
  gta_sa.run();

  // Clear old data
  // gta_sa.clear();
  return 0;
}
