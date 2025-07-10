// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef NEURON_DEFINITION_HPP
#define NEURON_DEFINITION_HPP

#include "statement.hpp"
#include "block.hpp"

class NeuronDefinition : public Statement
{
public:
    Block* code;

    NeuronDefinition(Block* B);
    void print(std::string indent) const override;
};

#endif
