#include "variable-definition.hpp"

#include <iostream>

void VariableDefinition::print(std::string indent) const
{
    declaration->print(indent);
    std::cout << " = ";
    expression->print(indent);
}
