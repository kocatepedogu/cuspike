#include "argument-list.hpp"

#include <iostream>

ArgumentList::ArgumentList(Expression *first)
{
    arguments.push_back(first);
}

ArgumentList::ArgumentList(Expression *first, Expression *second)
{
    arguments.push_back(first);
    arguments.push_back(second);
}

ArgumentList *ArgumentList::add(Expression *e)
{
    arguments.push_back(e);
    return this;
}

void ArgumentList::print(std::string indent) const
{
    std::cout << "(";
    for (Expression* v : arguments) {
        v->print("");
        std::cout << ",";
    }
    std::cout << ")";
}

bool ArgumentList::isSingle()
{
    return arguments.size() == 1;
}
