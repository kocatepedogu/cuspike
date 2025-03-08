#ifndef PARAMLIST_HPP
#define PARAMLIST_HPP

#include <vector>

#include "statement.hpp"
#include "variable-declaration.hpp"

class ParameterList : public Statement
{
private:
    std::vector<VariableDeclaration*> parameters;

public:
    ParameterList() = default;
    ParameterList(VariableDeclaration* first);
    ParameterList(VariableDeclaration* first, VariableDeclaration* second);

    ParameterList* add(VariableDeclaration* decl);
    void print(std::string indent) const override;
    bool isSingle();
};

#endif
