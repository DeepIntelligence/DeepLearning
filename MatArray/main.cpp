#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <armadillo>

#include "MatArray.h"

void loadData_MNIST(std::shared_ptr<arma::mat> X, 
                    std::shared_ptr<arma::mat> Y,
                    std::string filename_base);

int main(int argc, char *argv[]){
    MatArray<double>::Mat1DArray matArr = MatArray<double>::build(5);
    
    for (int i = 0 ; i < 5; i++) {
        matArr[0].randu(5,5);
        matArr[0].print();
    }
}


void loadData_MNIST(std::shared_ptr<arma::mat> X, 
                    std::shared_ptr<arma::mat> Y,
                    std::string filename_base){
  
  std::string filename;
  char tag[50];
  char x;
  int count;
  int numFiles = 10;
  int featSize = 28*28;
  int labelSize = 10;
  int numSamples = 10;
  X->set_size(numFiles*numSamples,featSize);
  Y->set_size(numFiles*numSamples,labelSize);
  Y->fill(0);
//  std::cout << Y.Len() << std::endl;
//  std::cout << X.NumR() << std::endl;
//  std::cout << X.NumC() << std::endl;
  
  for (int i = 0 ; i < numFiles ; i++){
    sprintf(tag,"%d",i);
    filename=filename_base+(std::string)tag;
    std::cout << filename << std::endl;
    std::ifstream infile;
    infile.open(filename,std::ios::binary | std::ios::in);
    if (infile.is_open()){
      
      for (int j = 0 ; j < numSamples ; j++){
        
        for (int k =0 ; k <featSize; k ++){
        infile.read(&x,1);
//        std::cout << x << std::endl;
        (*X)(j+i*numSamples,k)=(unsigned char)x;
        }
        (*Y)(j+i*numSamples,i) = 1;
      }
      
    } else{
      std::cout << "open file failure!" << std::endl;
    }

// for (int j = 0 ; j < numSamples ; j++){
//       for (int k =0 ; k <featSize; k ++){

//	           std::cout << x << std::endl;
//	   std::cout<<  (*X)(j,k) << " ";
//	   }
//	   }

    std::cout << "dataloading finish!" <<std::endl;

  }

}
