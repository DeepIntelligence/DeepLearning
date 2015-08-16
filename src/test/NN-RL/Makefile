include ../Makefile.common

all: test_NNRL

test_NNRL: NN_RL_Driver.o Model_PoleSimple.o Model_PoleFull.o Trainer.o NN_RLSolverBase.o NN_RLSolverSimple.o NN_RLSolverFull.o
	$(CXX) -o $@ $^ $(LDFLAG)

%.o:%.cpp
	$(CXX) -c $(CXXFLAGS) $^

Trainer.o: ../Trainer/Trainer.cpp
	$(CXX) -c $(CXXFLAGS) $^

clean:
	rm *.o
