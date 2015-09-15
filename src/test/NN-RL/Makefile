include ../Makefile.common

all: test_NNRL.exe

OBJ = NN_RL_Driver.o Model_PoleSimple.o Model_PoleFull.o Trainer.o NN_RLSolverBase.o NN_RLSolverMLP.o NN_RLSolverMultiMLP.o NN_RLSolverRNN.o RLSolver_2DTable.o

test_NNRL.exe: $(OBJ)
	$(CXX) -o $@ $^ $(LDFLAG) 





%.o:%.cpp
	$(CXX) -c $(CXXFLAGS) $(DEBUGFLAG) $^

Trainer.o: ../Trainer/Trainer.cpp
	$(CXX) -c $(CXXFLAGS) $^

clean:
	rm *.o *.exe
