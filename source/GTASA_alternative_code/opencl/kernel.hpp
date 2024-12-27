#ifndef JAMCRC_OPENCL_HPP
#define JAMCRC_OPENCL_HPP
#include <string>

namespace my::opencl::kernel {
std::string jamcrcTable();
std::string jamcrc1Byte();
std::string jamcrc1ByteLocal();
std::string jamcrc4Byte();
std::string generateStringLocal();
std::string cheatListTable();
std::string findAlternativeCheat();
}  // namespace my::opencl::kernel

#endif  // JAMCRC_OPENCL_HPP
