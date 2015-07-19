CPP_CUDA = nvcc
CPP = nvcc
CXXFLAGS = --std=c++11 -I/opt/boost_1_57_0
#nvcc mmul_1.cu -lcublas -lcurand -o mmul_1
LINKFLAGS = -lcublas -lcurand -L~/workspace/libs/gtest-1.7.0/mybuilds -lgtest

OBJ = test_GPUMat.o GPUMat.o GPU_Math_Func.o device_common.o

all: test 

test : $(OBJ)
	$(CPP) -o $@ $(OBJ) $(LINKFLAGS)
	
GPUMat.o : GPUMat.cpp
	$(CPP) -c $(CXXFLAGS) $@ $^

GPU_Math_Func.o : GPU_Math_Func.cu
	$(CPP_CUDA) -c $(CXXFLAGS) $@ $^
test_GPUMat.o : test_GPUMat.cpp
	$(CPP) -c $(CXXFLAGS) $@ $^
device_common.o : device_common.cpp
	$(CPP) -c $(CXXFLAGS) $< -o $@
#%.o : %.cpp
#	$(CPP) -c $(CXXFLAGS) 


clean:
	rm -f *.o *~ test
