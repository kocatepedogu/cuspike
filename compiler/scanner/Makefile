all: build/flex.o

build/flex.o: build/flex.c
	$(CXX) -std=c++14 -c build/flex.c -o build/flex.o
build/flex.c: scanner.l
	$(LEX) -o build/flex.c scanner.l

clean:
	rm build/*.o build/*.c
