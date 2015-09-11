
include ../Makefile.common

OBJ = test_RNN.o Trainer.o

test : $(OBJ)
	$(CXX) -o $@ $(OBJ) $(LDFLAG)
Trainer.o : ../Trainer/Trainer.cpp

	$(CXX) -c $(CXXFLAGS) $^
clean:
	rm -f *.o *~
