all: build/generator.o \
     build/typename-generator.o \
     build/expression-generator.o \
     build/statement-generator.o

build/generator.o: generator.cpp
	$(CXX) -c generator.cpp -o build/generator.o
build/typename-generator.o: typename-generator.cpp
	$(CXX) -c typename-generator.cpp -o build/typename-generator.o
build/expression-generator.o: expression-generator.cpp
	$(CXX) -c expression-generator.cpp -o build/expression-generator.o
build/statement-generator.o: statement-generator.cpp
	$(CXX) -c statement-generator.cpp -o build/statement-generator.o

clean:
	rm build/*.o

