// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef CONSTANT_HPP
#define CONSTANT_HPP

#include "expression.hpp"

class Constant : public Expression
{
public:
    std::string value;

    Constant(const std::string* v);
    Constant* clone() const override;

    void print(std::string indent) const override;
};

#endif
