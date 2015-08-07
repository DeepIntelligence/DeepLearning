DEEPLEARNING_PATH=-L/home/yuguangyang/Dropbox/DeepLearningPackage/YangCopy/DeepLearning/src/lib
DEEPLEARNING_INCLUDE=-I/home/yuguangyang/Dropbox/DeepLearningPackage/YangCopy/DeepLearning/include
MKL_INCLUDE = /opt/intel/composerxe/mkl/include

#CFLAGS = -std=c++0x -I/g/ssli/software/pkgs/boost_1_49_0 -c -DNDEBUG -D__LINUX $(OPTFLAGS) 
CPP = g++
ARMA_INCLUDE=-I/home/yuguangyang/Downloads/armadillo-5.100.2/include
ARMA_LINKFLAGS=-L/usr/lib -llapack -lblas
#MKLLINKFLAGS = -Wl,--start-group -Wl,--end-group -lpthread
CXXFLAGS = -std=c++0x $(ARMA_INCLUDE) $(DEEPLEARNING_INCLUDE) -I/opt/boost/boost_1_57_0 -c -D__LINUX -DDEBUG -g3 -DARMA_DONT_USE_WRAPPER -I/usr/local/include
LINKOPTFLAGS = -O3 -flto=4 -fwhole-program -march=native
#LINKFLAGS = -static $(LINKOPTFLAGS) $(MKLLINKFLAGS) -ldl
LINK_DL_FLAGS =   $(DEEPLEARNING_PATH) -L/usr/local/lib  $(ARMA_LINKFLAGS) -ldeeplearning -ldl #$(LINKOPTFLAGS)
#LINKFLAGS = 
#ODIR=obj
ODIR =

OBJ = main.o LSTMLayer.o

test : $(OBJ)
	$(CPP) -o test $(OBJ) $(LINK_DL_FLAGS)

LSTMLayer.o: LSTMLayer.cpp
	$(CPP) -c $(CXXFLAGS) $<
	

clean:
	rm -f *.o *~
