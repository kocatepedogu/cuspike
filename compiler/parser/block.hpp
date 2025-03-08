#ifndef BLOCK_HPP
#define BLOCK_HPP

#include <vector>

#include "statement.hpp"

class Block : public Statement
{
public:
    std::vector<Statement*> statements;

    Block() = default;
    Block(Statement *first);
    Block(Statement *first, Statement *second);

    Block *add(Statement *s);
    void print(std::string indent) const override;
    bool isSingle();
};

#endif
