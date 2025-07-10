// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef COMPOSITION_HPP
#define COMPOSITION_HPP

#include "expression.hpp"
#include "argument-list.hpp"

class Composition : public Expression
{
public:
    std::string* func;
    ArgumentList* args;

    Composition(std::string* f, ArgumentList* arg);
    Composition* clone() const override;

    void print(std::string indent) const override;
};

#endif
