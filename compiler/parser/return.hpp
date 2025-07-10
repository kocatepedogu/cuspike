// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef RETURN_HPP
#define RETURN_HPP

#include "statement.hpp"
#include "expression.hpp"

class Return : public Statement
{
public:
    Expression* return_value;

    Return(Expression* return_value);

    void print(std::string indent) const override;
};

#endif
