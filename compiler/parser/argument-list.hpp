// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef ARGLIST_HPP
#define ARGLIST_HPP

#include <vector>

#include "statement.hpp"
#include "expression.hpp"

class ArgumentList : public Statement
{
public:
    std::vector<Expression*> arguments;

    ArgumentList() = default;
    ArgumentList(Expression *first);
    ArgumentList(Expression *first, Expression *second);

    ArgumentList* add(Expression* e);

    void print(std::string indent) const override;

    bool isSingle();
};

#endif
