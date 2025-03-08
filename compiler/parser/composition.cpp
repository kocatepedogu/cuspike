#include "composition.hpp"

#include <iostream>

Composition::Composition(std::string* f, ArgumentList* arg) :
    Expression(COMPOSITION), func(f), args(arg)
{}

Composition* Composition::clone() const
{
    return new Composition(func, args);
}

void Composition::print(std::string indent) const
{
    std::cout << *func;
    ((Statement*)args)->print("");
}
