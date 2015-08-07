CC = g++
CFLAGS = -std=c++0x
LFLAGS = -L/usr/local/lib -lprotobuf
#all:vector_ser

test: test.o DeepLearning.pb.o
	$(CC) -o $@ $^ $(LFLAGS)

%.o : %.cpp
	$(CC) -c $(CFLAGS) $^

%.o : %.cc
	$(CC) -c $(CFLAGS) $^

#vector_ser.o:vector_ser.cpp
#	$(CC) -c $(CFLAGS) vector_ser.cpp


clean:
	rm *.o test
