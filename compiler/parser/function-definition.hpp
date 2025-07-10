// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef FUNCTION_DEFINITION_HPP
#define FUNCTION_DEFINITION_HPP

#include <string>

#include "words.hpp"
#include "statement.hpp"
#include "block.hpp"
#include "parameter-list.hpp"

class FunctionDefinition : public Statement
{
public:
    TypeName returnType;
    ParameterList* parameters;
    std::string name;
    Block* code;

    FunctionDefinition(TypeName T, std::string* N, ParameterList* P, Block* B);
    void print(std::string indent) const override;
};

#endif
