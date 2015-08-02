include ../Makefile.common

all: test_rnn_funcApprox

test_mlp: test_rnn_funcApprox.o MultiLayerPerceptron.o
	$(CXX) -o $@ $^ $(LDFLAG)

test_funcApprox: test_funcApprox.o MultiLayerPerceptron.o
	$(CXX) -o $@ $^ $(LDFLAG)

%.o:%.cpp
	$(CXX) -c $(CXXFLAGS) $^

clean:
	rm test_mlp *.o
