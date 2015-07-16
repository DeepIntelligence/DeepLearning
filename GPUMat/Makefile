CPP = nvcc
CXXFLAGS = -I/opt/boost_1_57_0
#nvcc mmul_1.cu -lcublas -lcurand -o mmul_1
LINKFLAGS = -lcublas -lcurand

OBJ = test_GPUMat.o GPUMat.o GPU_Math_Func.o

all: test test_static

test : $(OBJ)
	$(CPP) -o $@ $(OBJ) $(LINKFLAGS)
	
GPUMat.o : GPUMat.cpp
	$(CPP) -c $(CXXFLAGS) $@ $^

GPU_Math_Func.o : GPU_Math_Func.cu
	$(CPP) -c $(CXXFLAGS) $@ $^
test_GPUMat.o : test_GPUMat.cpp
	$(CPP) -c $(CXXFLAGS) $@ $^
#%.o : %.cpp
#	$(CPP) -c $(CXXFLAGS) 


clean:
	rm -f *.o *~
