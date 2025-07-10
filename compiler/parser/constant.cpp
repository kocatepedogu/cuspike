// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "constant.hpp"
#include <iostream>

Constant::Constant(const std::string* v)
    : Expression(VALUE), value(*v)
{}

Constant* Constant::clone() const
{
    return new Constant(&value);
}

void Constant::print(std::string indent) const
{
    std::cout << value;
}
