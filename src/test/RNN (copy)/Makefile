
include ../Makefile.common

OBJ = test_RNN.o Trainer.o RNN.o

test_RNN : $(OBJ)
	$(CXX) -o test $(OBJ) $(LDFLAG)
Trainer.o : ../Trainer/Trainer.cpp

	$(CXX) -c $(CXXFLAGS) $^
clean:
	rm -f *.o *~
