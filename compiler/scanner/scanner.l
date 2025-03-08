%{
#include "../../parser/build/bison.h"
void yyerror (const char *s);
%}

%option noyywrap

%%

int                                                     {yylval.type_name = INTEGER; return type;}
real                                                    {yylval.type_name = REAL; return type;}

Neuron                                                  {yylval.identifier = new std::string(yytext); return keyword_neuron;}
always                                                  {yylval.identifier = new std::string(yytext); return keyword_always;}
if                                                      {yylval.identifier = new std::string(yytext); return keyword_if;}
else                                                    {yylval.identifier = new std::string(yytext); return keyword_else;}

[._a-zA-Z][._a-zA-Z0-9]*                                {yylval.identifier = new std::string(yytext); return id;}
-?([0-9]+|[0-9]+\.[0-9]+)(e\-[0-9]+){0,1}(f|l){0,1}     {yylval.numeric_value = new std::string(yytext); return numeric;}

[ \t\n]                                                 {;}
[-+*/=;{}(),<>@]                                        {return yytext[0];}
.                                                       {ECHO; yyerror("unexpected character");}

%%

