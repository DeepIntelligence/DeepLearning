DEEPLEARNING_PATH=-L/home/kai/Dropbox/DeepLearningPackage/KaiCopy/DeepLearning/src/lib
DEEPLEARNING_INCLUDE=-I/home/kai/Dropbox/DeepLearningPackage/KaiCopy/DeepLearning/include
MKL_INCLUDE = /opt/intel/composerxe/mkl/include

#CFLAGS = -std=c++0x -I/g/ssli/software/pkgs/boost_1_49_0 -c -DNDEBUG -D__LINUX $(OPTFLAGS) 
CPP = g++
#MKLROOT = /opt/intel/mkl
#MKLROOT = /opt/intel/composer_xe_2013.0.079/mkl
MKLROOT = /opt/intel/composerxe/mkl
MKLLINKFLAGS = -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread
MKL_DL_LINKFLAGS =  -Wl,--no-as-needed -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm
ARMA_INCLUDE=-I~/Downloads/armadillo-5.100.2/include/armadillo
ARMA_LINKFLAGS=-L/usr/lib -L/opt/intel/mkl/lib/intel64 -larmadillo -lmkl_rt -llapack -lblas
#MKLLINKFLAGS = -Wl,--start-group -Wl,--end-group -lpthread
CXXFLAGS = -std=c++0x -I$(MKL_INCLUDE) $(ARMA_INCLUDE) $(DEEPLEARNING_INCLUDE) -I/opt/boost/boost_1_57_0 -c -D__LINUX -DDEBUG -g3
LINKOPTFLAGS = -O3 -flto=4 -fwhole-program -march=native
#LINKFLAGS = -static $(LINKOPTFLAGS) $(MKLLINKFLAGS) -ldl
LINK_DL_FLAGS =   $(DEEPLEARNING_PATH) $(MKL_DL_LINKFLAGS) $(ARMA_LINKFLAGS) -ldeeplearning -ldl -lprotobuf#$(LINKOPTFLAGS)
#LINKFLAGS = 
#ODIR=obj
ODIR =

OBJ = main.o BaseLayer_LSTM.o LSTMLayer.o

test : $(OBJ)
	$(CPP) -o test $(OBJ) $(LINK_DL_FLAGS)

BaseLayer_LSTM.o : BaseLayer_LSTM.cpp
	$(CPP) -c $(CXXFLAGS) $<

LSTMLayer.o: LSTMLayer.cpp
	$(CPP) -c $(CXXFLAGS) $<
	

clean:
	rm -f *.o *~
