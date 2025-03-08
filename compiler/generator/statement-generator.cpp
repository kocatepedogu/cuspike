#include "statement-generator.hpp"
#include "expression-generator.hpp"
#include "typename-generator.hpp"

#include <sstream>
#include <iostream>

static std::string regular_assignment_to_cuda_code(std::string lhs, std::string rhs, Operation op) {
    std::stringstream s;

    // Write left hand side
    s << lhs;

    // Write operator
    switch (op) {
        case NOP:
            s << " = ";
            break;
        case PLUS_EQ:
            s << " += ";
            break;
        case MINUS_EQ:
            s << " -= ";
            break;
        default:
            fprintf(stderr, "Internal error: Unhandled assignment type\n");
            abort();
    }

    // Write left hand side
    s << rhs;

    // End statement
    s << ";";

    // Return result
    return s.str();
}

static std::string atomic_assignment_to_cuda_code(std::string lhs, std::string rhs, Operation op) {
    std::stringstream s;

    // Write atomicAdd instructon and left hand side
    s << "atomicAdd(&" << lhs << ",";

    // Atomic operations must either involve + or -, direct assignment with = is not supported.
    if (op == NOP) {
        fprintf(stderr, "Atomic change to post-synaptic variable must be done with either += or -=");
        abort();
    }

    // Write right hand side
    if (op == MINUS_EQ) {
        s << "-(";
    }
    s << rhs;
    if (op == MINUS_EQ) {
        s << ")";
    }
    s << ");";

    // Return result
    return s.str();
}

std::string assignment_to_cuda_code(Assignment *assignment, KernelType kernel_type, bool atomic) {
    const Constant *lhs_constant = dynamic_cast<const Constant *>(assignment->lhs);
    if (!lhs_constant) {
        fprintf(stderr, "Left hand side of an assignment must be an identifier\n");
        abort();
    }

    std::string lhs = expression_to_cuda_code(assignment->lhs, kernel_type);
    std::string rhs = expression_to_cuda_code(assignment->rhs, kernel_type);

    if (is_neuron_symbol(lhs_constant->value) && kernel_type == GlobalKernel && atomic) {
        return atomic_assignment_to_cuda_code(lhs, rhs, assignment->op);
    }

    return regular_assignment_to_cuda_code(lhs, rhs, assignment->op);
}

std::string variable_definition_to_cuda_code(VariableDefinition *variable_definition, KernelType kernel_type, bool atomic) {
    std::stringstream s;

    // Get declaration part
    auto declaration = variable_definition->declaration;

    // Write type name
    s << typename_to_cuda_code(declaration->type) << " ";

    // Write variable name
    s << declaration->name;

    // Write assignment operator
    s << " = ";

    // Write definition part
    s << expression_to_cuda_code(variable_definition->expression, kernel_type);

    // End statement;
    s << ";";

    // Combine result
    return s.str();
}

std::string variable_declaration_to_cuda_code(VariableDeclaration *variable_declaration, KernelType kernel_type, bool atomic) {
    std::stringstream s;

    // Write type name
    s << typename_to_cuda_code(variable_declaration->type) << " ";

    // Write variable name
    s << variable_declaration->name;

    // End statement;
    s << ";";

    // Combine result
    return s.str();
}

std::string if_to_cuda_code(If *if_statement, KernelType kernel_type, bool atomic) {
    std::stringstream s;

    // Write if keyword and open parenthesis
    s << "if (";

    // Write condition
    s << expression_to_cuda_code(if_statement->condition, kernel_type);

    // Close parenthesis
    s << ")";

    // Write variable name
    s << statement_to_cuda_code(if_statement->body, kernel_type, atomic);

    // Write else part if exists
    if (if_statement->elseBody) {
        s << " else ";
        s << statement_to_cuda_code(if_statement->elseBody, kernel_type, atomic);
    }

    // Combine result
    return s.str();
}

std::string block_to_cuda_code(Block *block, KernelType kernel_type, bool atomic) {
    std::stringstream s;

    // Open scope
    s << "{" << std::endl;

    // Write all statements
    for (Statement *statement : block->statements) {
        s << statement_to_cuda_code(statement, kernel_type, atomic);
        s << std::endl;
    }

    // Close scope
    s << "}" << std::endl;

    // Combine result
    return s.str();
}

std::string statement_to_cuda_code(Statement *statement, KernelType kernel_type, bool atomic) {
    Assignment *assignment = dynamic_cast<Assignment *>(statement);
    if (assignment) {
        return assignment_to_cuda_code(assignment, kernel_type, atomic);
    }

    VariableDefinition *variable_definition = dynamic_cast<VariableDefinition *>(statement);
    if (variable_definition) {
        return variable_definition_to_cuda_code(variable_definition, kernel_type, atomic);
    }

    VariableDeclaration *variable_declaration = dynamic_cast<VariableDeclaration *>(statement);
    if (variable_declaration) {
        return variable_declaration_to_cuda_code(variable_declaration, kernel_type, atomic);
    }

    If *if_statement = dynamic_cast<If *>(statement);
    if (if_statement) {
        return if_to_cuda_code(if_statement, kernel_type, atomic);
    }

    Block *block_statement = dynamic_cast<Block *>(statement);
    if (block_statement) {
        return block_to_cuda_code(block_statement, kernel_type, atomic);
    }

    Expression *expr_statement = dynamic_cast<Expression *>(statement);
    if (expr_statement) {
        return expression_to_cuda_code(expr_statement, kernel_type);
    }

    fprintf(stderr, "Unknown statement type\n");
    abort();
}
