#ifndef EXPRESSION_GENERATOR_HPP
#define EXPRESSION_GENERATOR_HPP

#include "../parser/expression.hpp"
#include "../parser/composition.hpp"
#include "../parser/constant.hpp"
#include "../parser/binary-operation.hpp"

struct Symbol {
    std::string type;
    std::string name;
};

enum KernelType {
    RegisterKernel,
    GlobalKernel,
    Independent
};

bool is_constant_symbol(std::string name);
bool is_neuron_symbol(std::string name);

std::string constant_to_cuda_code(const Constant *constant, KernelType kernel_type);
std::string composition_to_cuda_code(const Composition *composition, KernelType kernel_type);
std::string binary_operation_to_cuda_code(const BinaryOperation *binary_operation, KernelType kernel_type);
std::string expression_to_cuda_code(const Expression *expression, KernelType kernel_type);

#endif
