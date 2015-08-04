
include ../Makefile.common

OBJ = test_RNN.o Trainer.o RNN.o BaseLayer_LSTM.o

test_RNN : $(OBJ)
	$(CXX) -o test $(OBJ) $(LDFLAG)
Trainer.o : ../Trainer/Trainer.cpp
	$(CXX) -c $(CXXFLAGS) $^
BaseLayer_LSTM.o : ../../../LSTM/BaseLayer_LSTM.cpp
	$(CXX) -c $(CXXFLAGS) $^
clean:
	rm -f *.o *~
