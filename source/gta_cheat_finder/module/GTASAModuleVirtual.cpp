#include "GTASAModuleVirtual.hpp"

GTASAModuleVirtual::GTASAModuleVirtual(COMPUTE_TYPE type) : _type(type) {}

GTASAModuleVirtual::~GTASAModuleVirtual() {}

COMPUTE_TYPE GTASAModuleVirtual::type() const {
    return _type;
}
