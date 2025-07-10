// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "expression-generator.hpp"

#include <sstream>

extern std::vector<Symbol> constantSymbols;
extern std::vector<Symbol> neuronSymbols;

bool is_constant_symbol(std::string name) {
    for (Symbol sym : constantSymbols) {
        if (sym.name == name) {
            return true;
        }
    }

    return false;
}

bool is_neuron_symbol(std::string name) {
    for (Symbol sym : neuronSymbols) {
        if (sym.name == name) {
            return true;
        }
    }

    return false;
}

std::string constant_to_cuda_code(const Constant *constant, KernelType kernel_type) {
    if (is_constant_symbol(constant->value)) {
        return constant->value;
    }

    if (is_neuron_symbol(constant->value)) {
        if (kernel_type == RegisterKernel) {
            return constant->value;
        }
        if (kernel_type == GlobalKernel) {
            return constant->value + "[i]";
        }

        fprintf(stderr, "Internal error: Unknown kernel type %d", kernel_type);
        abort();
    }

    return constant->value;
}

std::string composition_to_cuda_code(const Composition *composition, KernelType kernel_type) {
    std::stringstream s;

    // Write function name
    s << *composition->func;

    // Open left parenthesis
    s << "(";

    // Write function arguments
    auto arguments = composition->args->arguments;
    for (int i = 0; i < arguments.size(); ++i) {
        s << expression_to_cuda_code(arguments[i], kernel_type);

        if (i != arguments.size() - 1) {
            s << ",";
        }
    }

    // Close right parenthesis
    s << ")";

    // Combine result
    return s.str();
}

std::string binary_operation_to_cuda_code(const BinaryOperation *binary_operation, KernelType kernel_type) {
    #define CUDA_BINARY_OPERATION(op) \
    do { \
        std::stringstream s; \
        s << "("; \
        s << expression_to_cuda_code(op1, kernel_type); \
        s << ")"; \
        s << op; \
        s << "("; \
        s << expression_to_cuda_code(op2, kernel_type); \
        s << ")"; \
        return s.str(); \
    } while (0);

    const Expression *op1 = binary_operation->op1;
    const Expression *op2 = binary_operation->op2;

    switch (binary_operation->op)
    {
        case ADD:
            CUDA_BINARY_OPERATION("+");
        case SUB:
            CUDA_BINARY_OPERATION("-");
        case MUL:
            CUDA_BINARY_OPERATION("*");
        case DIV:
            CUDA_BINARY_OPERATION("/");

        case EQUAL:
            CUDA_BINARY_OPERATION("==");
        case NOT_EQUAL:
            CUDA_BINARY_OPERATION("!=");
        case GREATER:
            CUDA_BINARY_OPERATION(">");
        case GREATER_OR_EQUAL:
            CUDA_BINARY_OPERATION(">=");
        case LESS:
            CUDA_BINARY_OPERATION("<");
        case LESS_OR_EQUAL:
            CUDA_BINARY_OPERATION("<=");

        default:
            fprintf(stderr, "Internal error: Unhandled binary operation\n");
            abort();
            break;
    }
}

std::string expression_to_cuda_code(const Expression *expression, KernelType kernel_type) {
    const Constant *constant = dynamic_cast<const Constant *>(expression);
    if (constant) {
        return constant_to_cuda_code(constant, kernel_type);
    }

    const Composition *composition = dynamic_cast<const Composition *>(expression);
    if (composition) {
        return composition_to_cuda_code(composition, kernel_type);
    }

    const BinaryOperation *binary_operation = dynamic_cast<const BinaryOperation *>(expression);
    if (binary_operation) {
        return binary_operation_to_cuda_code(binary_operation, kernel_type);
    }

    fprintf(stderr, "Internal error: Unhandled subclass of expression\n");
    abort();
}
