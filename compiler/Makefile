all:
	cd parser && $(MAKE)
	cd scanner && $(MAKE)
	cd generator && $(MAKE)
	$(CXX) parser/build/*.o scanner/build/*.o generator/build/*.o -o compiler

clean:
	cd parser && $(MAKE) clean
	cd scanner && $(MAKE) clean
	cd generator && $(MAKE) clean


