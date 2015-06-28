
MKL_INCLUDE = /opt/intel/composerxe/mkl/include
CPP = g++
MKLROOT = /opt/intel/composerxe/mkl
MKLLINKFLAGS = -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread
MKL_DL_LINKFLAGS =  -Wl,--no-as-needed -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm
ARMA_INCLUDE=-I~/Downloads/armadillo-5.100.2/include/armadillo
ARMA_LINKFLAGS=-L/usr/lib -L/opt/intel/mkl/lib/intel64 -larmadillo -lmkl_rt -llapack -lopenblas
#MKLLINKFLAGS = -Wl,--start-group -Wl,--end-group -lpthread
CXXFLAGS = -std=c++0x -I$(MKL_INCLUDE) $(ARMA_INCLUDE) -I/opt/boost/boost_1_57_0 -c -DDEBUG -D__LINUX -g3 -I/home/yuguangyang/workspace/libs/gtest-1.7.0/include
LINKOPTFLAGS = -O3 -flto=4 -fwhole-program
#LINKFLAGS = -static $(LINKOPTFLAGS) $(MKLLINKFLAGS) $(ARMA_LINKFLAGS) -ldl
LINK_DL_FLAGS =  $(LINKOPTFLAGS) $(MKL_DL_LINKFLAGS) $(ARMA_LINKFLAGS) -ldl -pthread -L~/workspace/libs/gtest-1.7.0/mybuilds -lgtest
#LINKFLAGS = 
#ODIR=obj
ODIR =


OBJ = test_BaseLayer.o BaseLayer.o

test : $(OBJ)
	$(CPP) -o test $(OBJ) $(LINK_DL_FLAGS)
	
	
ARMA_INCLUDE=-I~/Downloads/armadillo-5.100.2/include -DARMA_DONT_USE_WRAPPER
ARMA_LINKFLAGS=-L/usr/lib -llapack -lopenblas -pthread	
LINKFLAGS = -static $(LINKOPTFLAGS) $(ARMA_LINKFLAGS)	-L~/workspace/libs/gtest-1.7.0/mybuilds -lgtest
test_static : $(OBJ)
	$(CPP) -o $@ $(OBJ) $(LINKFLAGS)
	

clean:
	rm -f *.o *~
