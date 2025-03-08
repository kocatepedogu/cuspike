#ifndef FUNCTION_DEFINITION_HPP
#define FUNCTION_DEFINITION_HPP

#include <string>

#include "words.hpp"
#include "statement.hpp"
#include "block.hpp"
#include "parameter-list.hpp"

class FunctionDefinition : public Statement
{
private:
    TypeName returnType;
    ParameterList* parameters;
    std::string name;
    Block* code;

public:
    FunctionDefinition(TypeName T, std::string* N, ParameterList* P, Block* B);
    void print(std::string indent) const override;
};

#endif
