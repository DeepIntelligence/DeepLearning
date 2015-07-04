#include <random>

namespace NeuralNet{
template<typename T>	
struct Random_Bernoulli{
//		std::random_device rd;
    std::mt19937 gen;
    std::bernoulli_distribution *d;
	    
    Random_Bernoulli(double p){
        d = new std::bernoulli_distribution(p);
    }
		
    double next(){
	if((*d)(gen)) return 1.0;
	else return 0.0;			
    }

    void modifier(T *p, int size){
        for (int i = 0; i < size; i++){
            // perform "drop"
            if((*d)(gen)) 
                *(p+i) = (T)(0);                   
        }
    }
}; 

	
	
	
	
}
