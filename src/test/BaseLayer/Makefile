include ../Makefile.common

test_BaseLayer: test_BaseLayer.o
	$(CXX) -o $@ $^ $(LDFLAG)

%.o:%.cpp
	$(CXX) -c $(CXXFLAGS) $^


clean:
	rm test_BaseLayer *.o
