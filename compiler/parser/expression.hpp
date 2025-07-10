// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef EXPRESSION_HPP
#define EXPRESSION_HPP

#include <string>

#include "statement.hpp"

enum Operation {
    ADD,
    SUB,
    MUL,
    DIV,
    PLUS_EQ,
    MINUS_EQ,

    EQUAL,
    NOT_EQUAL,
    GREATER,
    GREATER_OR_EQUAL,
    LESS,
    LESS_OR_EQUAL,

    COMPOSITION,
    VALUE,
    NOP
};

class ArgumentList;

class Expression : public Statement
{
protected:
    Expression(Operation oper) :
        op(oper) {}
public:
    Operation op;

    Expression() = default;
    Expression(const Expression& other);

    virtual void print(std::string indent) const = 0;
    virtual Expression* clone() const = 0;
};

#endif
