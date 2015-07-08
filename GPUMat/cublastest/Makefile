CPP = nvcc
CXXFLAGS = --std=c++11 -I/home/yuugangyang/Downloads/armadillo-5.200.2/include -DARMA_DONT_USE_WRAPPER
#nvcc mmul_1.cu -lcublas -lcurand -o mmul_1
LINKFLAGS = -lcublas -lcurand -lblas -llapack

OBJ = main.o 

all: test 

test : $(OBJ)
	$(CPP) -o $@ $(OBJ) $(LINKFLAGS)
	
main.o : main.cpp
	$(CPP) -c $(CXXFLAGS) $@ $^
	
GPUMat.o : GPUMat.cpp
	$(CPP) -c $@ $^

#%.o : %.cpp
#	$(CPP) -c $(CXXFLAGS) 


clean:
	rm -f *.o *~
