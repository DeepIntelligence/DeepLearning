include ../Makefile.common

//all: test_funcApprox

#test_mlp: test_mlp.o MultiLayerPerceptron.o
#	$(CXX) -o $@ $^ $(LDFLAG)

//test_funcApprox: test_funcApprox.o MultiLayerPerceptron.o
//	$(CXX) -o $@ $^ $(LDFLAG)

%.o:%.cpp
	$(CXX) -c $(CXXFLAGS) $^


clean:
	rm *.o
