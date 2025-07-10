// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef FUNCTION_GENERATOR_HPP
#define FUNCTION_GENERATOR_HPP

#include "../parser/function-definition.hpp"
#include "expression-generator.hpp"

#include <string>

std::string function_to_cuda_code(const FunctionDefinition *function, KernelType kernel_type);

#endif
