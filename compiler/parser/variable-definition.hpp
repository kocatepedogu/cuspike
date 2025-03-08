#ifndef VARIABLEDEF_HPP
#define VARIABLEDEF_HPP

#include <string>

#include "statement.hpp"
#include "variable-declaration.hpp"
#include "expression.hpp"

class VariableDefinition : public Statement
{
public:
    VariableDeclaration* declaration;
    Expression* expression;

    VariableDefinition(VariableDeclaration* decl, Expression* e) :
        declaration(new VariableDeclaration(*decl)), expression(e) {}

    void print(std::string indent) const override;
};

#endif
