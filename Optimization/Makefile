CPP = g++
ARMA_INCLUDE=-I~/Downloads/armadillo-5.100.2/include
ARMA_LINKFLAGS= -lblas -llapack
CXXFLAGS = -std=c++0x -I$(ARMA_INCLUDE)  -c -DARMA_DONT_USE_WRAPPER  
LINKOPTFLAGS = -O3 -flto=4 -fwhole-program
LINKFLAGS = -static $(LINKOPTFLAGS)
LINK_DL_FLAGS =  $(MKL_DL_LINKFLAGS) $(ARMA_LINKFLAGS) -ldl



OBJ = main.o LBFGS.o SteepDescent.o

test : $(OBJ)
	$(CPP) -o test $(OBJ) $(LINKFLAGS)
	
clean:
	rm -f *.o *~
