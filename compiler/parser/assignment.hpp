// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef ASSIGNMENT_HPP
#define ASSIGNMENT_HPP

#include "statement.hpp"
#include "expression.hpp"

class Assignment : public Statement
{
public:
    const Expression* lhs;
    const Expression* rhs;

    const Operation op;

    Assignment(const Expression* l, const Expression* r) :
        lhs(l), rhs(r), op(NOP) {}

    Assignment(const Expression* l, const Expression* r, const Operation o) :
        lhs(l), rhs(r), op(o) {}

    void print(std::string indent) const override;
};

#endif
