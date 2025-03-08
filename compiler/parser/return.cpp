#include "return.hpp"

#include <iostream>

Return::Return(Expression* return_value)
{
    this->return_value = return_value;
}

void Return::print(std::string indent) const
{
    std::cout << indent << "return ";
    this->return_value->print("");
    std::cout << ";" << std::endl;
}
