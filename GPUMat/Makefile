CPP = nvcc
CXXFLAGS = 
#nvcc mmul_1.cu -lcublas -lcurand -o mmul_1
LINKFLAGS = -lcublas -lcurand

OBJ = main.o GPUMat.o 

all: test test_static

test : $(OBJ)
	$(CPP) -o $@ $(OBJ) $(LINKFLAGS)
	
main.o : main.cu
	$(CPP) -c $@ $^
	
GPUMat.o : GPUMat.cu
	$(CPP) -c $@ $^

#%.o : %.cpp
#	$(CPP) -c $(CXXFLAGS) 


clean:
	rm -f *.o *~