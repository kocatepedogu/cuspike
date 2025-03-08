#include "generator.hpp"
#include "expression-generator.hpp"
#include "typename-generator.hpp"
#include "statement-generator.hpp"
#include "function-generator.hpp"

#include "../parser/parser.hpp"

#include <fstream>
#include <iostream>

extern std::vector<Statement *> statements;

std::vector<Symbol> constantSymbols;
std::vector<Symbol> neuronSymbols;

static NeuronDefinition *getNeuronDefinition() {
    NeuronDefinition *result = nullptr;

    int count = 0;
    for (Statement* st : statements)
    {
        NeuronDefinition *neuron_definition_st = dynamic_cast<NeuronDefinition *>(st);
        if (neuron_definition_st) {
            result = neuron_definition_st;
            ++count;
        }
    }

    if (count == 0) {
        fprintf(stderr, "No neuron definition was provided.\n");
        abort();
    }

    if (count >= 2) {
        fprintf(stderr, "There cannot be more than one neuron definition.\n");
        abort();
    }

    return result;
}

template <typename Func>
static void forEachVariableDefinition(std::vector<Statement *> statements, Func func) {
    for (Statement* st : statements)
    {
        VariableDefinition *variable_definition_st = dynamic_cast<VariableDefinition *>(st);
        if (variable_definition_st)
        {
            auto declaration = variable_definition_st->declaration;
            auto expression = variable_definition_st->expression;

            auto type = declaration->type;
            auto name = declaration->name;

            func(typename_to_cuda_code(type), name, expression);
        }
    }
}

static void processConstantSymbols() {
    std::ofstream config_file("./generated/config.hpp");

    forEachVariableDefinition(statements, [&](std::string type, std::string name, Expression *expr) {
        config_file << "constexpr " << type << " " << name << " = " << expression_to_cuda_code(expr, Independent) << ";" << std::endl;
        constantSymbols.push_back(Symbol{type, name});
    });
}

static void processNeuronSymbols(NeuronDefinition *neuron_definition) {
    std::ofstream kernel_register_initialize("./generated/kernel-register/initialize.cu");
    std::ofstream kernel_global_initialize("./generated/kernel-global/initialize.cu");
    std::ofstream kernel_global_device_arrays("./generated/kernel-global/device-arrays.cu");

    forEachVariableDefinition(neuron_definition->code->statements,
        [&](std::string type, std::string name, Expression *expr) {
            kernel_register_initialize << type << " " << name << " = " << expression_to_cuda_code(expr, RegisterKernel) << ";" << std::endl;
            kernel_global_initialize << name << "[i] = " << expression_to_cuda_code(expr, GlobalKernel) << ";" << std::endl;
            kernel_global_device_arrays << "__device__ " << type << " " << name << "[N];" << std::endl;

            neuronSymbols.push_back(Symbol{type, name});
    });
}

void processAlwaysBlocks(NeuronDefinition *neuron_definition) {
    std::ofstream kernel_register_always("./generated/kernel-register/always.cu");
    std::ofstream kernel_register_always_at_active("./generated/kernel-register/always-at-active.cu");
    std::ofstream kernel_register_always_at_reset("./generated/kernel-register/always-at-reset.cu");
    std::ofstream kernel_register_always_at_spike("./generated/kernel-register/always-at-spike.cu");

    std::ofstream kernel_global_always("./generated/kernel-global/always.cu");
    std::ofstream kernel_global_always_at_active("./generated/kernel-global/always-at-active.cu");
    std::ofstream kernel_global_always_at_reset("./generated/kernel-global/always-at-reset.cu");
    std::ofstream kernel_global_always_at_spike("./generated/kernel-global/always-at-spike.cu");

    for (Statement* st : neuron_definition->code->statements)
    {
        Always *always_st = dynamic_cast<Always *>(st);
        if (always_st)
        {
            for (Statement *inner_st : always_st->body->statements) {
                if (always_st->condition == "*")
                {
                    kernel_register_always << statement_to_cuda_code(inner_st, RegisterKernel, false) << std::endl;
                    kernel_global_always << statement_to_cuda_code(inner_st, GlobalKernel, false) << std::endl;
                }
                else if (always_st->condition == "reset")
                {
                    kernel_register_always_at_reset << statement_to_cuda_code(inner_st, RegisterKernel, false) << std::endl;
                    kernel_global_always_at_reset << statement_to_cuda_code(inner_st, GlobalKernel, false) << std::endl;
                }
                else if (always_st->condition == "active")
                {
                    kernel_register_always_at_active << statement_to_cuda_code(inner_st, RegisterKernel, false) << std::endl;
                    kernel_global_always_at_active << statement_to_cuda_code(inner_st, GlobalKernel, false) << std::endl;
                }
                else if (always_st->condition == "spike")
                {
                    kernel_register_always_at_spike << statement_to_cuda_code(inner_st, RegisterKernel, true) << std::endl;
                    kernel_global_always_at_spike << statement_to_cuda_code(inner_st, GlobalKernel, true) << std::endl;
                }
                else
                {
                    fprintf(stderr, "Unknown always condition: %s\n", always_st->condition.c_str());
                    abort();
                }
            }
        }
    }
}

void processFunctions(NeuronDefinition *neuron_definition) {
    std::ofstream kernel_register_device_functions("./generated/kernel-register/device_functions.cu");
    std::ofstream kernel_global_device_functions("./generated/kernel-global/device_functions.cu");

    for (Statement* st : neuron_definition->code->statements)
    {
        FunctionDefinition *function_st = dynamic_cast<FunctionDefinition *>(st);
        if (function_st) {
            kernel_register_device_functions << function_to_cuda_code(function_st, RegisterKernel) << std::endl;
            kernel_global_device_functions << function_to_cuda_code(function_st, GlobalKernel) << std::endl;
        }
    }
}

void generator() {
    NeuronDefinition *neuron_definition = getNeuronDefinition();

    processConstantSymbols();
    processNeuronSymbols(neuron_definition);
    processAlwaysBlocks(neuron_definition);
    processFunctions(neuron_definition);
}
