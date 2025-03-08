#include "variable-declaration.hpp"

#include <iostream>

VariableDeclaration::VariableDeclaration(TypeName T, std::string* n)
{
    this->type = T;
    this->name = *n;
}

VariableDeclaration::VariableDeclaration(const VariableDeclaration& other)
{
    this->type = other.type;
    this->name = other.name;
}

void VariableDeclaration::print(std::string indent) const
{
    std::cout << indent;

    if (type == REAL) {
        std::cout << "real";
    }

    if (type == INTEGER) {
        std::cout << "int";
    }

    std::cout << " " << name;
}
