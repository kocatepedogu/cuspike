#ifndef VARIABLEDECL_HPP
#define VARIABLEDECL_HPP

#include <string>

#include "statement.hpp"
#include "words.hpp"

class VariableDeclaration : public Statement
{
public:
    TypeName type;
    std::string name;

    VariableDeclaration(TypeName T, std::string* n);
    VariableDeclaration(const VariableDeclaration& other);

    void print(std::string indent) const override;
};

#endif
