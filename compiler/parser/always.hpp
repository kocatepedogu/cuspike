// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef ALWAYS_HPP
#define ALWAYS_HPP

#include "statement.hpp"
#include "block.hpp"

class Always : public Statement
{
public:
    std::string condition;
    Block* body;

    Always(Block* body);
    Always(Block* body, std::string *condition);

    void print(std::string indent) const override;
};

#endif
