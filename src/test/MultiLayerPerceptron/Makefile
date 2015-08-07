include ../Makefile.common

test_mlp: test_funcApprox.o Trainer.o
	$(CXX) -o $@ $^ $(LDFLAG)
%.o:%.cpp
	$(CXX) -c $(CXXFLAGS) $^
Trainer.o:../Trainer/Trainer.cpp
	$(CXX) -c $(CXXFLAGS) $^

clean:
	rm *.o
