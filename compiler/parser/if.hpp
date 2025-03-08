#ifndef IF_HPP
#define IF_HPP

#include "statement.hpp"
#include "expression.hpp"

class If : public Statement
{
public:
    Expression* condition;
    Statement* body;
    Statement* elseBody;

    If(Expression* condition, Statement* body);
    If(Expression* condition, Statement* body, Statement *elseBody);

    void print(std::string indent) const override;
};

#endif
