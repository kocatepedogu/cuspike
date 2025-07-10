// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "function-generator.hpp"
#include "typename-generator.hpp"
#include "expression-generator.hpp"
#include "statement-generator.hpp"

#include <sstream>

std::string function_to_cuda_code(const FunctionDefinition *function, KernelType kernel_type) {
    std::stringstream s;

    s << std::endl << "__device__ inline" << std::endl;
    s << typename_to_cuda_code(function->returnType);
    s << " ";
    s << function->name;
    s << "(";

    for (int i = 0; i < function->parameters->parameters.size(); ++i) {
        auto variable_declaration = function->parameters->parameters[i];
        auto type = variable_declaration->type;
        auto name = variable_declaration->name;

        s << typename_to_cuda_code(type);
        s << " ";
        s << name;

        if (i != function->parameters->parameters.size() - 1) {
            s << ",";
        }
    }

    s << ")" << std::endl;
    s << block_to_cuda_code(function->code, kernel_type, false);

    return s.str();
}
