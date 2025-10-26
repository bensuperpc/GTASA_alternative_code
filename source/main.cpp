#include <cstdint>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>


int main(int argc, char* argv[]) {
#if ENABLE_QT == 1
#endif
#if ENABLE_OPENMP == 1
#endif
#if ENABLE_CUDA == 1
#endif
#if ENABLE_OPENCL == 1
#endif
    /*
    bool cli_only = false;

    std::ios_base::sync_with_stdio(false);  // Improve std::cout speed

    std::vector<std::string> args(argv + 1, argv + argc);

    uint64_t calc_mode = 0;
    uint64_t minRange = 0;
    uint64_t maxRange = 0;

    for (auto i = args.begin(); i != args.end(); ++i) {
        if (*i == "-h" || *i == "--help") {
            std::cout << "Syntax: GTASA_alternative_code --min <from (uint64_t)> --max "
                         "<to (uint64_t)>"
                         "--calc-mode <0-2> 0: std::thread, 1: OpenMP, 2: CUDA>"
                      << std::endl;
            return EXIT_SUCCESS;
        }
        if (*i == "--min") {
            std::istringstream iss(*++i);
            if (!(iss >> minRange)) {
                std::cout << "Error, non-numeric character !" << std::endl;
                return EXIT_FAILURE;
            }
        } else if (*i == "--max") {
            std::istringstream iss(*++i);
            if (!(iss >> maxRange)) {
                std::cout << "Error, non-numeric character !" << std::endl;
                return EXIT_FAILURE;
            }
        } else if (*i == "--calc-mode") {
            std::istringstream iss(*++i);
            if (!(iss >> calc_mode)) {
                std::cout << "Error, non-numeric character !" << std::endl;
                return EXIT_FAILURE;
            }
        } else if (*i == "--cli") {
            cli_only = true;
        } else {
            std::cout << "Unknown argument: " << *i << std::endl;
        }
    }

    std::unique_ptr<GTA_SA_ENGINE> gta_sa_main = std::make_unique<GTA_SA_ENGINE>();

    if (gta_sa_main == nullptr) {
        std::cout << "Error, gtaSA == nullptr" << std::endl;
        return EXIT_FAILURE;
    }

    switch (calc_mode) {
        case 0: {
            gta_sa_main->swichMode(COMPUTE_TYPE::STDTHREAD);
            break;
        }
        case 1: {
            gta_sa_main->swichMode(COMPUTE_TYPE::OPENMP);
            break;
        }
        case 2: {
#ifdef BUILD_WITH_CUDA
            gta_sa_main->swichMode(COMPUTE_TYPE::CUDA);
#else
            std::cout << "CUDA not supported, switching to STDTHREAD" << std::endl;
            gta_sa_main->swichMode(COMPUTE_TYPE::STDTHREAD);
#endif
            break;
        }
        case 3: {
#ifdef BUILD_WITH_OPENCL
            gta_sa_main->swichMode(COMPUTE_TYPE::OPENCL);
#else
            std::cout << "OPENCL not supported, switching to STDTHREAD" << std::endl;
            gta_sa_main->swichMode(COMPUTE_TYPE::STDTHREAD);
#endif
            break;
        }
        default: {
            std::cout << "Unknown calc mode: " << calc_mode << std::endl;
            break;
        }
    }
*/
        // gtaSA->threadCount
        // gtaSA->cudaBlockSize
    return 0;
}
