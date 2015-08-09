include ../Makefile.common

all: test_NNRL

test_NNRL: NN_RL_Driver.o Model_PoleSimple.o Trainer.o NN_RLSolver.o
	$(CXX) -o $@ $^ $(LDFLAG)

%.o:%.cpp
	$(CXX) -c $(CXXFLAGS) $^

Trainer.o: ../Trainer/Trainer.cpp
	$(CXX) -c $(CXXFLAGS) $^

clean:
	rm *.o
