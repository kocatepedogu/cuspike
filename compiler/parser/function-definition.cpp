// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "function-definition.hpp"

#include <iostream>

FunctionDefinition::FunctionDefinition(
    TypeName T, std::string* N, ParameterList* P, Block* B)
{
    returnType = T;
    parameters = P;
    name = *N;
    code = B;
}

void FunctionDefinition::print(std::string indent) const
{
    std::cout << std::endl;
    std::cout << indent;

    if (returnType == REAL) {
        std::cout << "real";
    }

    if (returnType == INTEGER) {
        std::cout << "int";
    }

    std::cout << " " << name;
    parameters->print("");
    std::cout << std::endl;

    std::cout << indent << "{" << std::endl;
    code->print(indent + "  ");
    std::cout << indent << "}";
}
