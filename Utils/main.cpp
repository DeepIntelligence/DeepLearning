#include <iostream>
#include "Util.h"


using namespace NeuralNet;
	
int main(){	
	
	Random_Bernoulli r(0.5);
	
	for(int i = 0; i < 100; i++)
		std::cout << r.next() << std::endl;	
	
	return 0;	
}	
	
