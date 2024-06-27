#include "GTA_SA_cheat_finder_result.hpp"

result::result(std::uint64_t _index, std::string _code, std::uint32_t _jamcrc, std::string _associated_code)
    : index(_index), code(_code), jamcrc(_jamcrc), associated_code(_associated_code) {}

result::~result() {}

result& result::operator=(const result& other) noexcept {
    this->index = other.index;
    this->code = other.code;
    this->jamcrc = other.jamcrc;
    this->associated_code = other.associated_code;

    return *this;
}

bool result::operator==(const result& other) const noexcept {
    return (this->index == other.index && this->code == other.code && this->jamcrc == other.jamcrc);
}
