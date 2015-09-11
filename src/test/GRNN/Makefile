include ../Makefile.common

test_MALayer: test_MALayer.o MultiAddLayer.o
	$(CXX) -o $@ $^ $(LDFLAG)
%.o:%.cpp
	$(CXX) -c $(CXXFLAGS) $^


clean:
	rm *.o
