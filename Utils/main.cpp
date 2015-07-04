#include <iostream>
#include "Util.h"


using namespace NeuralNet;
	
int main(){	
	
	Random_Bernoulli<double> r(0.5);
	
	for(int i = 0; i < 100; i++)
		std::cout << r.next() << std::endl;


        Random_Bernoulli<int> r2(0.5);
	
        int p[25];
        for(int i = 0; i < 25; i++ ){
            p[i] = 1;
        }
        
        r2.modifier(p,25);
        std::cout << "second" << std::endl;
        for(int i = 0; i < 25; i++)
            std::cout << p[i] << std::endl;
        
        
        Random_Bernoulli<unsigned long long> r3(0.5);
	
	unsigned long long p2[25];
        for(int i = 0; i < 25; i++ ){
            p2[i] = 1;
        }
        
        r3.modifier(p2,25);
        std::cout << "third" << std::endl;
        for(int i = 0; i < 25; i++)
            std::cout << p2[i] << std::endl;
	return 0;	
}	
	
