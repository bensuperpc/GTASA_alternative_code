#include "GTA_SA_cheat_finder.hpp"

GTA_SA::GTA_SA() {}

void GTA_SA::clear()
{
  this->results.clear();
  this->results.shrink_to_fit();
}

void GTA_SA::run()
{
#if !defined(_OPENMP)
  if (this->calc_mode == 1) {
    std::cout << "OpenMP is not enabled, please compile with -fopenmp flag"
                 "or select another calculation mode (CUDA or std::thread), "
                 "fall back to std::thread."
                 "Doesn't impact performance most of the time."
              << std::endl;
    this->calc_mode = 0;
  }
#endif

#if !defined(BUILD_WITH_CUDA)
  if (this->calc_mode == 2) {
    std::cout << "CUDA is not enabled, please compile with CUDA"
                 "or select another calculation mode (OpenMP or std::thread), "
                 "fall back to std::thread."
                 "Less performant than CUDA."
              << std::endl;
    this->calc_mode = 0;
  }
#endif
  if (this->calc_mode == 3) {
    std::cout << "OpenCL is not ready, please select another calculation mode "
                 "(OpenMP, std::thread), fall back to std::thread."
              << std::endl;
    this->calc_mode = 0;
  }

  if (this->calc_mode > 3) {
    std::cout << "Invalid calculation mode, fall back to std::thread."
              << std::endl;
    this->calc_mode = 0;
  }

  if (this->calc_mode == 0) {
    std::cout << "Running with std::thread mode" << std::endl;
  } else if (this->calc_mode == 1) {
    std::cout << "Running with OpenMP mode" << std::endl;
  } else if (this->calc_mode == 2) {
    std::cout << "Running with CUDA mode" << std::endl;
  } else if (this->calc_mode == 3) {
    std::cout << "Running with OpenCL mode" << std::endl;
  } else {
    std::cout << "Unknown calculation mode" << std::endl;
  }

  std::cout << "Max thread support: " << this->max_thread_support()
            << std::endl;
  std::cout << "Running with: " << this->num_thread << " threads" << std::endl;

  std::array<char, 29> tmp1 = {0};
  std::array<char, 29> tmp2 = {0};

  if (this->min_range > this->max_range) {
    std::cout << "Min range value: '" << this->min_range
              << "' can't be greater than Max range value: '" << this->max_range
              << "'" << std::endl;
    return;
  }

  if ((this->max_range - this->min_range) < 1) {
    std::cout << "Search range is too small." << std::endl;
    return;
  }

  std::cout << "Number of calculations: " << (this->max_range - this->min_range)
            << std::endl;

  GTA_SA::find_string_inv(tmp1.data(), this->min_range);
  GTA_SA::find_string_inv(tmp2.data(), this->max_range);
  std::cout << "From: " << tmp1.data() << " to: " << tmp2.data()
            << " Alphabetic sequence" << std::endl;
  this->begin_time = std::chrono::high_resolution_clock::now();
  if (this->calc_mode == 0) {
    BS::thread_pool pool(this->num_thread);
    pool.parallelize_loop(
        min_range,
        max_range,
        [&](const std::uint64_t& _min_range, const std::uint64_t& _max_range)
        {
          for (std::uint64_t i = _min_range; i <= _max_range; i++) {
            runner(i);
          }
        });
  } else if (this->calc_mode == 1) {
#if defined(_OPENMP)
    omp_set_num_threads(static_cast<int>(num_thread));
#endif

#if defined(_OPENMP)
#  ifdef _MSC_VER
    std::int64_t i =
        0;  // OpenMP (2.0) on Windows doesn't support unsigned variable
#    pragma omp parallel for shared(results) schedule(dynamic)
// firstprivate(tmp, crc)
#  else
    std::uint64_t i = 0;
#    pragma omp parallel for schedule(auto) shared(results)
// firstprivate(tmp, crc)
#  endif
    for (i = min_range; i <= max_range; i++) {
      runner(i);
    }
#else
    std::cout << "OpenMP is not supported" << std::endl;
#endif
  } else if (this->calc_mode == 2) {
#if defined(BUILD_WITH_CUDA)

    if ((max_range - min_range) < cuda_block_size) {
      std::cout << "Number of calculations is less than cuda_block_size"
                << std::endl;
    }

    std::vector<uint32_t> jamcrc_results;
    std::vector<uint64_t> index_results;

    my::cuda::launch_kernel(jamcrc_results,
                            index_results,
                            min_range,
                            max_range,
                            this->cuda_block_size);

    for (uint64_t i = 0; i < jamcrc_results.size(); ++i) {
      std::array<char, 29> tmpCUDA = {0};

      GTA_SA::find_string_inv(tmpCUDA.data(), index_results[i]);
      std::reverse(
          tmpCUDA.data(),
          tmpCUDA.data() + strlen(tmpCUDA.data()));  // Invert char array

#  if ((defined(_MSVC_LANG) && _MSVC_LANG >= 202002L) \
       || __cplusplus >= 202002L && !defined(ANDROID) \
           && !defined(__EMSCRIPTEN__) && !defined(__clang__))

      const auto&& it = std::find(std::execution::unseq,
                                  std::begin(GTA_SA::cheat_list),
                                  std::end(GTA_SA::cheat_list),
                                  jamcrc_results[i]);
#  else
      const auto&& it = std::find(std::begin(GTA_SA::cheat_list),
                                  std::end(GTA_SA::cheat_list),
                                  jamcrc_results[i]);
#  endif

      const uint64_t index =
          static_cast<uint64_t>(it - std::begin(GTA_SA::cheat_list));
      results.emplace_back(std::make_tuple(index_results[i],
                                           std::string(tmpCUDA.data()),
                                           jamcrc_results[i],
                                           cheat_list_name.at(index)));
    }
#else
    std::cout << "CUDA is not supported" << std::endl;
#endif
  } else if (this->calc_mode == 3) {
  } else {
    std::cout << "Unknown calculation mode" << std::endl;
  }

  this->end_time = std::chrono::high_resolution_clock::now();
  std::cout << "" << std::endl;

  std::sort(this->results.begin(), this->results.end());  // Sort results

  constexpr auto display_val = 18;

  std::cout << std::setw(display_val + 3) << "Iter. N°"
            << std::setw(display_val) << "Code" << std::setw(display_val + 8)
            << "JAMCRC value" << std::endl;

  for (auto& result : this->results) {
    std::cout << std::setw(display_val + 3) << std::get<0>(result)
              << std::setw(display_val) << std::get<1>(result)
              << std::setw(display_val) << "0x" << std::hex
              << std::get<2>(result) << std::dec << std::endl;
  }
  std::cout << "Time: "
            << std::chrono::duration_cast<std::chrono::duration<double>>(
                   this->end_time - this->begin_time)
                   .count()
            << " sec" << std::endl;  // Display time

  std::cout << "This program execute: " << std::fixed
            << (static_cast<double>(this->max_range - this->min_range)
                / std::chrono::duration_cast<std::chrono::duration<double>>(
                      this->end_time - this->begin_time)
                      .count())
          / 1000000
            << " MOps/sec" << std::endl;  // Display perf
  std::cout << "" << std::endl;
}

void GTA_SA::runner(const std::uint64_t& i)
{
  std::array<char, 29> tmp = {0};
  GTA_SA::find_string_inv(tmp.data(),
                          i);  // Generate Alphabetic sequence from uint64_t
                               // value, A=1, Z=27, AA = 28, AB = 29
  uint32_t crc = GTA_SA::jamcrc(tmp.data());  // JAMCRC

  // #pragma omp critical
  // std::cout << "str:" << tmp.data() << " crc: " << crc << std::endl;

#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 202002L) \
     || __cplusplus >= 202002L && !defined(ANDROID) \
         && !defined(__EMSCRIPTEN__) && !defined(__clang__))

  const auto&& it = std::find(std::execution::unseq,
                              std::begin(GTA_SA::cheat_list),
                              std::end(GTA_SA::cheat_list),
                              crc);
#else
  const auto&& it = std::find(
      std::begin(GTA_SA::cheat_list), std::end(GTA_SA::cheat_list), crc);
#endif

  // If crc is present in Array
  if (it != std::end(GTA_SA::cheat_list)) {
    std::reverse(tmp.data(),
                 tmp.data() + strlen(tmp.data()));  // Invert char array

    const uint64_t index =
        static_cast<uint64_t>(it - std::begin(GTA_SA::cheat_list));

    this->results.emplace_back(std::make_tuple(
        i,
        std::string(tmp.data()),
        crc,
        cheat_list_name.at(index)));  // Save result: calculation position,
                                      // Alphabetic sequence, CRC,
  }
}

#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus >= 201703L)
auto GTA_SA::jamcrc(std::string_view my_string, const uint32_t previousCrc32)
    -> std::uint32_t
{
#else

#  if _MSC_VER && !__INTEL_COMPILER
#    pragma message( \
        "C++17 is not enabled, the program will be less efficient with previous standards")
#  else
#    warning C++17 is not enabled, the program will be less efficient with previous standards.
#  endif

auto GTA_SA::jamcrc(const std::string& my_string, const uint32_t previousCrc32)
    -> std::uint32_t
{
#endif
  auto crc = ~previousCrc32;
  const auto* current =
      reinterpret_cast<const unsigned char*>(my_string.data());
  uint64_t length = my_string.length();
  // process eight bytes at once
  while (static_cast<bool>(length--)) {
    crc = (crc >> 8) ^ crc32_lookup[(crc & 0xFF) ^ *current++];
  }
  return crc;
}

/**
 * \brief Generate Alphabetic sequence from uint64_t value, A=0, Z=26, AA = 27,
 * T \param n index in base 26 \param array return array
 */
void GTA_SA::find_string_inv(char* array, uint64_t n)
{
  constexpr std::uint32_t string_size_alphabet {27};
  constexpr std::array<char, string_size_alphabet> alpha {
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"};
  // If n < 27
  if (n < 26) {
    array[0] = alpha[n];
    return;
  }
  // If n > 27
  std::uint64_t i = 0;
  while (n > 0) {
    array[i] = alpha[(--n) % 26];
    n /= 26;
    ++i;
  }
}
