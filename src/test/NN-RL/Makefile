include ../Makefile.common

all: test_NNRL

test_NNRL: NN_RL_Driver.o model.o
	$(CXX) -o $@ $^ $(LDFLAG)

%.o:%.cpp
	$(CXX) -c $(CXXFLAGS) $^


clean:
	rm *.o
