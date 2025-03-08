#include "if.hpp"

#include <iostream>

If::If(Expression* condition, Statement* body)
{
    this->condition = condition;
    this->body = body;
    this->elseBody = nullptr;
}

If::If(Expression* condition, Statement* body, Statement *elseBody)
{
    this->condition = condition;
    this->body = body;
    this->elseBody = elseBody;
}

void If::print(std::string indent) const
{
    std::cout << indent << "if (";
    this->condition->print("");
    std::cout << ")";

    std::cout << std::endl;
    std::cout << indent << "{" << std::endl;
    this->body->print(indent + "  ");
    std::cout << indent << "}" << std::endl;

    if (this->elseBody) {
        std::cout << indent << "else";

        std::cout << std::endl;
        std::cout << indent << "{" << std::endl;
        this->elseBody->print(indent + "  ");
        std::cout << indent << "}" << std::endl;
    }
}
