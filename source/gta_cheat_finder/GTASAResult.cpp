#include "GTASAResult.hpp"

GTASAResult::GTASAResult(std::uint64_t _index, std::string _code, std::uint32_t _jamcrc, std::string _associated_code)
    : index(_index), code(_code), jamcrc(_jamcrc), associated_code(_associated_code) {}

GTASAResult::GTASAResult() {}

GTASAResult::~GTASAResult() {}

GTASAResult& GTASAResult::operator=(const GTASAResult& other) noexcept {
    this->index = other.index;
    this->code = other.code;
    this->jamcrc = other.jamcrc;
    this->associated_code = other.associated_code;

    return *this;
}

bool GTASAResult::operator==(const GTASAResult& other) const noexcept {
    return (this->index == other.index && this->code == other.code && this->jamcrc == other.jamcrc);
}
