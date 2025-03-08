#include "block.hpp"

#include <iostream>

Block::Block(Statement *first)
{
    statements.push_back(first);
}

Block::Block(Statement *first, Statement *second)
{
    statements.push_back(first);
    statements.push_back(second);
}

Block *Block::add(Statement *s)
{
    statements.push_back(s);
    return this;
}

void Block::print(std::string indent) const
{
    for (Statement* s : statements) {
        s->print(indent + "  ");
        std::cout << std::endl;
    }
}

bool Block::isSingle()
{
    return statements.size() == 1;
}
