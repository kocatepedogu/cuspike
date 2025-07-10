// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef STATEMENT_HPP
#define STATEMENT_HPP

#include <string>

class Statement
{
public:
    Statement() = default;
    virtual void print(std::string indent) const = 0;
};

#endif
