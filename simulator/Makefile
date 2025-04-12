NVCC=nvcc
NVCC_FLAGS=-arch=native -Xcompiler="-fopenmp -O3"

all: simulator.exe

simulator.exe: main.o initialize.o save.o kernel-register.o kernel-global.o
	$(NVCC) $(NVCC_FLAGS) *.o -o simulator.exe

main.o: main.cu
	$(NVCC) $(NVCC_FLAGS) -c main.cu -o main.o

initialize.o: initialize.cu
	$(NVCC) $(NVCC_FLAGS) -c initialize.cu -o initialize.o

save.o: save.cu
	$(NVCC) $(NVCC_FLAGS) -c save.cu -o save.o

kernel-register.o: kernel-register.cu
	$(NVCC) $(NVCC_FLAGS) -c kernel-register.cu -o kernel-register.o

kernel-global.o: kernel-global.cu
	$(NVCC) $(NVCC_FLAGS) -c kernel-global.cu -o kernel-global.o

clean:
	rm -rf *.o
	rm -rf simulator.exe

