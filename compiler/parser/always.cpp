#include "always.hpp"

#include <iostream>

Always::Always(Block *body)
{
    this->condition = "*";
    this->body = body;
}

Always::Always(Block *body, std::string *condition)
{
    this->condition = *condition;
    this->body = body;
}

void Always::print(std::string indent) const
{
    std::cout << std::endl;

    if (this->condition == "*") {
        std::cout << indent << "always";
    } else {
        std::cout << indent << "always @(" << this->condition << ")";
    }

    std::cout << std::endl;

    std::cout << indent << "{" << std::endl;
    body->print(indent + "  ");
    std::cout << indent << "}";
}
