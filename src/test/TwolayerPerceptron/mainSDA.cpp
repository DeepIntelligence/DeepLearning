#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <armadillo>
#include "MultiLayerPerceptron.h"




void loadData_MNIST(std::shared_ptr<arma::mat> X,
                    std::shared_ptr<arma::mat> Y);

int main(int argc, char *argv[]) {
    std::shared_ptr<arma::mat> trainDataX(new arma::mat);
    std::shared_ptr<arma::mat> trainDataY(new arma::mat);
    loadData_MNIST(trainDataX,trainDataY);

    int inputDim = trainDataX->n_cols;
    int outputDim = trainDataY->n_cols;
    int hiddenDim = 100;
    std::cout << inputDim << std::endl;
    std::cout << outputDim << std::endl;
    std::cout << trainDataX->n_rows << std::endl;
    std::cout << trainDataY->n_rows << std::endl;
    trainDataX->save("trainingSamples.txt",arma::raw_ascii);
    TrainingPara trainingPara(1e-6,100, 10, 0.5);
    trainingPara.print();
    MultiLayerPerceptron mlp(inputDim,  outputDim, hiddenDim, trainDataX, trainDataY, trainingPara);

    mlp.train();

    mlp.test(trainDataX,trainDataY);
// after training i do some testing

}


void loadData_MNIST(std::shared_ptr<arma::mat> X,
                    std::shared_ptr<arma::mat> Y) {

    std::string filename_base("../MNIST/data");
    std::string filename;
    char tag[50];
    char x;
    int count;
    int numFiles = 10;
    int featSize = 28*28;
    int labelSize = 10;
    int numSamples = 100;
    X->set_size(numFiles*numSamples,featSize);
    Y->set_size(numFiles*numSamples,labelSize);
    Y->fill(0);
//  std::cout << Y.Len() << std::endl;
//  std::cout << X.NumR() << std::endl;
//  std::cout << X.NumC() << std::endl;

    for (int i = 0 ; i < numFiles ; i++) {
        sprintf(tag,"%d",i);
        filename=filename_base+(std::string)tag;
        std::cout << filename << std::endl;
        std::ifstream infile;
        infile.open(filename,std::ios::binary | std::ios::in);
        if (infile.is_open()) {

            for (int j = 0 ; j < numSamples ; j++) {

                for (int k =0 ; k <featSize; k ++) {
                    infile.read(&x,1);
                    (*X)(j+i*numSamples,k)=(unsigned char)x;
                    (*X)(j+i*numSamples,k)/= 256.0;
                }
                (*Y)(j+i*numSamples,i) = 1;
            }

        } else {
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