#pragma once
#include <random>
#include <string>
#include <memory>

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

class RandomStream{
private:
    std::shared_ptr<std::mt19937> genPtr;
    std::shared_ptr<std::uniform_real_distribution<>> randomPtr_unitformReal;
    std::shared_ptr<std::uniform_int_distribution<>> randomPtr_unitformInt; 
public:     
    RandomStream(){
        
        std::random_device rd;
        genPtr = std::make_shared<std::mt19937>(rd());
        randomPtr_unitformReal = std::make_shared<std::uniform_real_distribution<>>(0.0, 1.0);
    }
    RandomStream(int low , int high){
        
        std::random_device rd;
        genPtr = std::make_shared<std::mt19937>(rd());
        
        randomPtr_unitformReal = std::make_shared<std::uniform_real_distribution<>>(0.0, 1.0);
        randomPtr_unitformInt = std::make_shared<std::uniform_int_distribution<>>(low, high);
    }
    double nextDou(){return (*randomPtr_unitformReal)(*genPtr);}
    int nextInt(){return (*randomPtr_unitformInt)(*genPtr);}
};	
	
void loadData_MNIST(std::shared_ptr<arma::mat> X,
                    std::shared_ptr<arma::mat> Y, std::string);
	
	
}
