// SPDX-FileCopyrightText: 2025 Doğu Kocatepe
// SPDX-License-Identifier: GPL-3.0-or-later

#include "binary-operation.hpp"
#include "expression.hpp"

#include <iostream>
#include <sstream>

BinaryOperation::BinaryOperation(const Expression* e1, const Expression* e2, Operation oper) :
    Expression(oper), op1(e1), op2(e2) {}

BinaryOperation* BinaryOperation::clone() const
{
    return new BinaryOperation(op1, op2, op);
}

void BinaryOperation::print(std::string indent) const
{
#define PRINT_BINARY_OPERATION(op) \
    std::cout << "("; \
    op1->print(""); \
    std::cout << ")"; \
    std::cout << op; \
    std::cout << "("; \
    op2->print(""); \
    std::cout << ")"; \
    return;

    switch (op)
    {
        case ADD:
            PRINT_BINARY_OPERATION("+");
        case SUB:
            PRINT_BINARY_OPERATION("-");
        case MUL:
            PRINT_BINARY_OPERATION("*");
        case DIV:
            PRINT_BINARY_OPERATION("/");

        case EQUAL:
            PRINT_BINARY_OPERATION("==");
        case NOT_EQUAL:
            PRINT_BINARY_OPERATION("!=");
        case GREATER:
            PRINT_BINARY_OPERATION(">");
        case GREATER_OR_EQUAL:
            PRINT_BINARY_OPERATION(">=");
        case LESS:
            PRINT_BINARY_OPERATION("<");
        case LESS_OR_EQUAL:
            PRINT_BINARY_OPERATION("<=");

        default:
            perror("Internal error at BinaryOperation::print: Unhandled binary operation\n");
            abort();
            break;
    }
}

