#ifndef JAMCRC_OPENCL_HPP
#define JAMCRC_OPENCL_HPP
#include <string>

namespace my::opencl::kernel
{
    std::string jamcrc_table();
    std::string jamcrc1Byte();
    std::string jamcrc4Byte();
    std::string generateString();
}

#endif  // JAMCRC_OPENCL_HPP
