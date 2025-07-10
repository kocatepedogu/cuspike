// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "assignment.hpp"
#include "expression.hpp"

#include <iostream>

void Assignment::print(std::string indent) const
{
    std::cout << indent;
    lhs->print("");

    switch (op) {
        case NOP:
            std::cout << " = ";
            break;
        case PLUS_EQ:
            std::cout << " += ";
            break;
        case MINUS_EQ:
            std::cout << " -= ";
            break;
        default:
            perror("Internal error: Unhandled assignment statement");
            abort();
            break;
    }

    rhs->print("");
}
