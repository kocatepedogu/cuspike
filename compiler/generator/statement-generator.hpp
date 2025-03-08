#ifndef STATEMENT_GENERATOR_HPP
#define STATEMENT_GENERATOR_HPP

#include "expression-generator.hpp"

#include "../parser/block.hpp"
#include "../parser/assignment.hpp"
#include "../parser/variable-definition.hpp"
#include "../parser/variable-declaration.hpp"
#include "../parser/if.hpp"

std::string assignment_to_cuda_code(Assignment *assignment, KernelType kernel_type, bool atomic);
std::string variable_definition_to_cuda_code(VariableDefinition *variable_definition, KernelType kernel_type, bool atomic);
std::string variable_declaration_to_cuda_code(VariableDeclaration *variable_declaration, KernelType kernel_type, bool atomic);
std::string if_to_cuda_code(If *if_statement, KernelType kernel_type, bool atomic);
std::string block_to_cuda_code(Block *block, KernelType kernel_type, bool atomic);
std::string statement_to_cuda_code(Statement *statement, KernelType kernel_type, bool atomic);

#endif
