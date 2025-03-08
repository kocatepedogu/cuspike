#ifndef BINARY_OPERATION_HPP
#define BINARY_OPERATION_HPP

#include "expression.hpp"
#include "constant.hpp"

class BinaryOperation : public Expression
{
public:
    const Expression* op1;
    const Expression* op2;

    BinaryOperation(const Expression* e1, const Expression* e2, Operation oper);
    BinaryOperation* clone() const override;

    void print(std::string indent) const override;
};

static inline BinaryOperation operator * (const Expression& e1, const Expression& e2) {
    return BinaryOperation(&e1, &e2, MUL);
}

static inline BinaryOperation operator / (const Expression& e1, const Expression& e2) {
    return BinaryOperation(&e1, &e2, DIV);
}

static inline BinaryOperation operator + (const Expression& e1, const Expression& e2) {
    return BinaryOperation(&e1, &e2, ADD);
}

static inline BinaryOperation operator - (const Expression& e1, const Expression& e2) {
    return BinaryOperation(&e1, &e2, SUB);
}

static inline BinaryOperation operator == (const Expression& e1, const Expression& e2) {
    return BinaryOperation(&e1, &e2, EQUAL);
}

static inline BinaryOperation operator != (const Expression& e1, const Expression& e2) {
    return BinaryOperation(&e1, &e2, NOT_EQUAL);
}

static inline BinaryOperation operator > (const Expression& e1, const Expression& e2) {
    return BinaryOperation(&e1, &e2, GREATER);
}

static inline BinaryOperation operator >= (const Expression& e1, const Expression& e2) {
    return BinaryOperation(&e1, &e2, GREATER_OR_EQUAL);
}

static inline BinaryOperation operator < (const Expression& e1, const Expression& e2) {
    return BinaryOperation(&e1, &e2, LESS);
}

static inline BinaryOperation operator <= (const Expression& e1, const Expression& e2) {
    return BinaryOperation(&e1, &e2, LESS_OR_EQUAL);
}

static inline BinaryOperation operator - (const Expression& e) {
    return *new Constant(new std::string("0")) - e;
}

#endif
