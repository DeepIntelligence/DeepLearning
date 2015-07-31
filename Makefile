CPP = g++
ARMA_INCLUDE=-I/home/yuguangyang/Downloads/armadillo-5.100.2/include
ARMA_LINKFLAGS=-llapack -lblas
CXXFLAGS = -std=c++0x $(ARMA_INCLUDE) -I./include -I/opt/boost/boost_1_57_0 -c -D__LINUX -DDEBUG -g3 -DARMA_DONT_USE_WRAAPER

OBJ = BaseLayer.o

# Specify extensions of files to delete when cleaning
CLEANEXTS   = o a 

# Specify the target file and the install directory
OUTPUTFILE  = libdeeplearning.a
INSTALLDIR  = src/lib

$(OUTPUTFILE) : %.o
	ar ru $@ $^
	ranlib $@

%.o : src/%.cpp
	$(CPP) -c $(CXXFLAGS) $^
	

clean:
	for file in $(CLEANEXTS); do rm -f *.$$file; done
	
install:
	mkdir -p $(INSTALLDIR)
	cp -p $(OUTPUTFILE) $(INSTALLDIR)
