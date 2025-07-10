// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <vector>

#include "expression.hpp"
#include "statement.hpp"

std::vector<Statement *> statements;

Expression::Expression(const Expression& other) :
    op(other.op)
{}

