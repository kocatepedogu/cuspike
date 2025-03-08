%{

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <iostream>
#include <ostream>
#include <cstring>
#include <vector>
#include <memory>

void yyerror (const char *s);
int yyparse();
int yylex(void);

%}

%code requires {
    #include <vector>
    #include "../parser.hpp"
    #include "../../generator/generator.hpp"

    extern std::vector<Statement *> statements;
}

%union {
    Statement *statement;
    Expression *expr;
    Block *block;
    VariableDeclaration *variabledeclaration;
    VariableDefinition *variabledefinition;
    ArgumentList *argumentlist;
    ParameterList *parameterlist;
    FunctionDefinition *functiondefinition;
    NeuronDefinition *neurondefinition;
    Always *always;
    If *ifexpr;

    std::string *keyword;
    std::string *identifier;
    std::string *numeric_value;
    TypeName type_name;
}

%start line

%token <numeric_value> numeric
%token <identifier> id
%token <type_name> type

%token <keyword> keyword_neuron
%token <keyword> keyword_always
%token <keyword> keyword_if
%token <keyword> keyword_else

%type <expr> exp term factor func_comp
%type <statement> statement expression_st assignment_st
%type <block> block_st_c block_st
%type <variabledeclaration> variable_dec_st
%type <variabledefinition> variable_def_st
%type <parameterlist> param_list_st param_list_st_c
%type <argumentlist> arg_list_st arg_list_st_c
%type <functiondefinition> function_def_st
%type <neurondefinition> neuron_def_st
%type <always> always_st
%type <ifexpr> ifexpr_st

%%

line            : statement ';'                             {statements.push_back($1);}
                | line statement ';'                        {statements.push_back($2);}
                ;

statement       : expression_st                             {$$ = $1;}
                | assignment_st                             {$$ = $1;}
                | block_st                                  {$$ = $1;}
                | variable_dec_st                           {$$ = $1;}
                | variable_def_st                           {$$ = $1;}
                | function_def_st                           {$$ = $1;}
                | neuron_def_st                             {$$ = $1;}
                | always_st                                 {$$ = $1;}
                | ifexpr_st                                 {$$ = $1;}
                ;

expression_st   : exp                                       {$$ = $1;}
                ;

assignment_st   : exp '=' exp                               {$$ = new Assignment($1, $3);}
                | exp '+' '=' exp                           {$$ = new Assignment($1, $4, PLUS_EQ);}
                | exp '-' '=' exp                           {$$ = new Assignment($1, $4, MINUS_EQ);}
                ;

block_st        : '{' '}'                                   {$$ = new Block;}
                | '{' block_st_c '}'                        {$$ = $2;}
                ;
block_st_c      : statement ';'                             {$$ = new Block($1);}
                | block_st_c statement ';'                  {$$ = $1->add($2);}
                ;

exp             : term                                      {$$ = $1;}
                | exp '=' '=' exp                           {$$ = (*$1 == *$4).clone();}
                | exp '!' '=' exp                           {$$ = (*$1 != *$4).clone();}
                | exp '>' exp                               {$$ = (*$1 > *$3).clone();}
                | exp '>' '=' exp                           {$$ = (*$1 >= *$4).clone();}
                | exp '<' exp                               {$$ = (*$1 < *$3).clone();}
                | exp '<' '=' exp                           {$$ = (*$1 <= *$4).clone();}
                | exp '+' term                              {$$ = (*$1 + *$3).clone();}
                | exp '-' term                              {$$ = (*$1 - *$3).clone();}
                ;

term            : factor                                    {$$ = $1;}
                | term '*' factor                           {$$ = (*$1 * *$3).clone();}
                | term '/' factor                           {$$ = (*$1 / *$3).clone();}
                ;

factor          : id                                        {$$ = new Constant($1);}
                | numeric                                   {$$ = new Constant($1);}
                | func_comp                                 {$$ = $1;}
                | '(' exp ')'                               {$$ = $2;}
                ;

variable_dec_st : type id                                   {$$ = new VariableDeclaration($1, $2);}
                ;

variable_def_st : variable_dec_st '=' exp                   {$$ = new VariableDefinition($1, $3);}
                ;

param_list_st   : '(' variable_dec_st ')'                   {$$ = new ParameterList($2);}
                | '(' param_list_st_c variable_dec_st ')'   {$$ = $2->add($3);}
                ;
param_list_st_c : variable_dec_st ','                       {$$ = new ParameterList($1);}
                | param_list_st_c variable_dec_st ','       {$$ = $1->add($2);}
                ;

arg_list_st     : '(' exp ')'                               {$$ = new ArgumentList($2);}
                | '(' arg_list_st_c exp ')'                 {$$ = $2->add($3);}
                ;
arg_list_st_c   : exp ','                                   {$$ = new ArgumentList($1);}
                | arg_list_st_c exp ','                     {$$ = $1->add($2);}
                ;

function_def_st : type id '(' ')' block_st                  {$$ = new FunctionDefinition($1, $2, new ParameterList, $5);}
                | type id param_list_st block_st            {$$ = new FunctionDefinition($1, $2, $3, $4);}
                ;

func_comp       : id '(' ')'                                {$$ = new Composition($1, new ArgumentList);}
                | id arg_list_st                            {$$ = new Composition($1, $2);}
                ;

neuron_def_st   : keyword_neuron block_st                   {$$ = new NeuronDefinition($2);}
                ;

always_st       : keyword_always '@' '(' id ')' block_st    {$$ = new Always($6, $4);}
                | keyword_always block_st                   {$$ = new Always($2);}
                ;

ifexpr_st       : keyword_if '(' exp ')' statement          {$$ = new If($3, $5);}
                | keyword_if '(' exp ')' statement
                    keyword_else statement                  {$$ = new If($3, $5, $7);}
                ;

%%

int main(void)
{
    yyparse();

    for (Statement* st : statements)
    {
        st->print("");
        std::cout << std::endl;
    }

    generator();
}

void yyerror(const char *s)
{
    fprintf(stderr, "%s\n", s);
}
