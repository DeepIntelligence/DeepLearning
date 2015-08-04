
include ../Makefile.common

OBJ = test_trainer.o Trainer.o MultiLayerPerceptron.o

test_IO : $(OBJ)
	$(CXX) -o test $(OBJ) $(LDFLAG)
MultiLayerPerceptron.o : ../MultiLayerPerceptron/MultiLayerPerceptron.cpp
	$(CXX) -c $(CXXFLAGS) $^
clean:
	rm -f *.o *~
