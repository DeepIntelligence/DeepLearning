CPP = g++

ARMA_INCLUDE=-I~/Downloads/armadillo-5.100.2/include -DARMA_DONT_USE_WRAPPER
ARMA_LINKFLAGS=-L/usr/lib -llapack -lopenblas -pthread
# the following line is the sucessful one: some observations: armadillo can do the matrix calculation using only -larmadillo
#ARMA_DL_LINKFLAGS= -L/opt/intel/mkl/lib/intel64 -larmadillo -lmkl_rt -llapack -lblas
ARMA_DL_LINKFLAGS= -llapack -lopenblas
#ARMA_DL_INCLUDE=-I~/Downloads/armadillo-5.100.2/include/armadillo
CXXFLAGS = -std=c++0x $(ARMA_INCLUDE)
LINKOPTFLAGS = -O3 -flto=4 -fwhole-program
LINKFLAGS = -static $(LINKOPTFLAGS) $(ARMA_LINKFLAGS)
LINK_DL_FLAGS =   $(MKL_DL_LINKFLAGS) $(ARMA_DL_LINKFLAGS) -ldl
#LINKFLAGS = 
#ODIR=obj
ODIR =

OBJ = example1.o

all: test test_static

test : $(OBJ)
	$(CPP) -o test $(OBJ) $(LINK_DL_FLAGS)
	
	
test_static: $(OBJ)
	$(CPP) -o test_static $(OBJ) $(LINKFLAGS)
	
#%.o : %.cpp
#	$(CPP) -c $(CXXFLAGS) 


clean:
	rm -f *.o *~



