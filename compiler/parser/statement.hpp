#ifndef STATEMENT_HPP
#define STATEMENT_HPP

#include <string>

class Statement
{
public:
    Statement() = default;
    virtual void print(std::string indent) const = 0;
};

#endif
