#include "typename-generator.hpp"

std::string typename_to_cuda_code(const TypeName type) {
    switch (type) {
        case REAL:
            return "float";
        case INTEGER:
            return "uint32_t";
        default:
            fprintf(stderr, "Internal error at typename_to_code: Unhandled type name\n");
            abort();
    }
}
