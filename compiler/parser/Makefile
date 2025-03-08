YACC=bison

all: build/bison.o \
     build/argument-list.o \
     build/assignment.o \
     build/block.o \
     build/expression.o \
     build/constant.o \
     build/composition.o \
     build/binary-operation.o \
     build/function-definition.o \
     build/neuron-definition.o \
     build/always.o \
     build/if.o \
     build/return.o \
     build/parameter-list.o \
     build/variable-declaration.o \
     build/variable-definition.o

build/bison.o: build/bison.c
	$(CXX) -c build/bison.c -o build/bison.o

build/bison.c: parser.y words.hpp
	$(YACC) -d parser.y -o build/bison.c #-Wcounterexamples

build/argument-list.o: argument-list.cpp argument-list.hpp
	$(CXX) -c argument-list.cpp -o build/argument-list.o

build/assignment.o: assignment.cpp assignment.hpp
	$(CXX) -c assignment.cpp -o build/assignment.o

build/block.o: block.cpp block.hpp
	$(CXX) -c block.cpp  -o build/block.o

build/expression.o: expression.cpp expression.hpp
	$(CXX) -c expression.cpp  -o build/expression.o

build/binary-operation.o: binary-operation.cpp binary-operation.hpp
	$(CXX) -c binary-operation.cpp  -o build/binary-operation.o

build/constant.o: constant.cpp constant.hpp
	$(CXX) -c constant.cpp  -o build/constant.o

build/composition.o: composition.cpp composition.hpp
	$(CXX) -c composition.cpp  -o build/composition.o

build/function-definition.o: function-definition.cpp function-definition.hpp
	$(CXX) -c function-definition.cpp  -o build/function-definition.o

build/neuron-definition.o: neuron-definition.cpp neuron-definition.hpp
	$(CXX) -c neuron-definition.cpp  -o build/neuron-definition.o

build/always.o: always.cpp always.hpp
	$(CXX) -c always.cpp  -o build/always.o

build/if.o: if.cpp if.hpp
	$(CXX) -c if.cpp  -o build/if.o

build/return.o: return.cpp return.hpp
	$(CXX) -c return.cpp  -o build/return.o

build/parameter-list.o: parameter-list.cpp parameter-list.hpp
	$(CXX) -c parameter-list.cpp  -o build/parameter-list.o

build/variable-declaration.o: variable-declaration.cpp variable-declaration.hpp
	$(CXX) -c variable-declaration.cpp  -o build/variable-declaration.o

build/variable-definition.o: variable-definition.cpp variable-definition.hpp
	$(CXX) -c variable-definition.cpp  -o build/variable-definition.o

clean:
	rm build/*.o
	rm build/*.c
	rm build/*.h

