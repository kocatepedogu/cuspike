// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "neuron-definition.hpp"

#include <iostream>

NeuronDefinition::NeuronDefinition(Block* B)
{
    code = B;
}

void NeuronDefinition::print(std::string indent) const
{
    std::cout << std::endl;
    std::cout << indent << "Neuron";
    std::cout << std::endl;

    std::cout << indent << "{" << std::endl;
    code->print(indent + "  ");
    std::cout << indent << "}" << std::endl;
}
