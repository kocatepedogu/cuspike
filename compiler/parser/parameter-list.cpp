#include "parameter-list.hpp"

#include <iostream>

ParameterList::ParameterList(VariableDeclaration* first)
{
    parameters.push_back(new VariableDeclaration(*first));
}

ParameterList::ParameterList(
    VariableDeclaration* first, VariableDeclaration* second)
{
    parameters.push_back(new VariableDeclaration(*first));
    parameters.push_back(new VariableDeclaration(*second));
}

ParameterList* ParameterList::add(VariableDeclaration* decl)
{
    parameters.push_back(new VariableDeclaration(*decl));
    return this;
}

void ParameterList::print(std::string indent) const
{
    std::cout << "(";
    for (VariableDeclaration* v : parameters) {
        v->print("");
        std::cout << ",";
    }
    std::cout << ")";
}

bool ParameterList::isSingle()
{
    return parameters.size() == 1;
}
