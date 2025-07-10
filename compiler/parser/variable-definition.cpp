// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "variable-definition.hpp"

#include <iostream>

void VariableDefinition::print(std::string indent) const
{
    declaration->print(indent);
    std::cout << " = ";
    expression->print(indent);
}
