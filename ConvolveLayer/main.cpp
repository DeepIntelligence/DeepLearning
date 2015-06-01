#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <armadillo>
#include <vector>
#include "ConvolveLayer.h"


void loadData_MNIST(std::shared_ptr<arma::mat> X, 
                    std::shared_ptr<arma::mat> Y);

int main(int argc, char *argv[]){
  std::shared_ptr<arma::mat> DataX(new arma::mat);
  std::shared_ptr<arma::mat> DataY(new arma::mat);
  std::shared_ptr<arma::mat> trainDataX(new arma::mat);
  std::shared_ptr<arma::mat> trainDataY(new arma::mat);
  std::shared_ptr<arma::mat> testDataX(new arma::mat);
  std::shared_ptr<arma::mat> testDataY(new arma::mat);
  std::shared_ptr<arma::mat> ValidationDataX(new arma::mat);
  std::shared_ptr<arma::mat> ValidationDataY(new arma::mat);
    
  loadData_MNIST(DataX,DataY);
  
  int ntrain = 1000;
  int ntest = 100;
//  now I split data into train, test, and validation  
  trainDataX = std::make_shared<arma::mat>(DataX->rows(0,ntrain-1));
  trainDataY = std::make_shared<arma::mat>(DataY->rows(0,ntrain-1));
  testDataX = std::make_shared<arma::mat>(DataX->rows(ntrain,ntrain+ntest-1));
  testDataY = std::make_shared<arma::mat>(DataY->rows(ntrain,ntrain+ntest-1));
  
  DataX.reset();
  DataY.reset();
    
  

  int inputDim = trainDataX->n_cols;
  int outputDim = trainDataY->n_cols;
  std::cout << inputDim << std::endl;
  std::cout << outputDim << std::endl;
  std::cout << trainDataX->n_rows << std::endl;
  std::cout << trainDataY->n_rows << std::endl;
  

  int numLayers = 2;
  std::vector<int> dimensions;
  
  dimensions.push_back(784);
  dimensions.push_back(100);
  dimensions.push_back(50);
  
 
}


void loadData_MNIST(std::shared_ptr<arma::mat> X, 
                    std::shared_ptr<arma::mat> Y){
  
  std::string filename_base("../MNIST/data");
  std::string filename;
  char tag[50];
  char x;
  int count;
  int numFiles = 10;
  int featSize = 28*28;
  int labelSize = 10;
  int numSamples = 1000;
  X->set_size(numFiles*numSamples,featSize);
  Y->set_size(numFiles*numSamples,labelSize);
  Y->fill(0);

  
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
        (*X)(i+numFiles*j,k)=((unsigned char)x)/256.0;
        
        }
        (*Y)(i+numFiles*j,i) = 1;
//        count++;
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
