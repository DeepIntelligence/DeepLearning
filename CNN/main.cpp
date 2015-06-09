#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <armadillo>
#include <vector>
#include "CNN.h"
#include "../MatArray/MatArray.h"

void loadData_MNIST(std::shared_ptr<arma::mat> X,
                    std::shared_ptr<arma::mat> Y);

void load_cifar10(std::shared_ptr<arma::cube> &X,
                    std::shared_ptr<arma::mat> &Y,
                    std::string filename_base, int numFiles,
                    int numSamples);

void workOnMNIST();
void workOnCIFAR10();

int main(int argc, char *argv[]) {

//    workOnMNIST();
    
    workOnCIFAR10();
}

void workOnCIFAR10(){

std::string filename = "../cifar-10/cifar-10-batches-bin/data_batch_";

std::shared_ptr<arma::cube> X;

std::shared_ptr<arma::mat> Y;

load_cifar10(X, Y, filename,1,500);
// first try to plot some image
for (int i = 0; i < 1 ; i++){
    arma::mat temp = X->slice(i);
    temp.reshape(1,32*32);
    temp.save("image.dat", arma::raw_ascii);

}


TrainingPara trainingPara(1e-6,100, 1, 0.1);

   CNN cnn(X, Y,3, std::move(trainingPara));
   cnn.train();
   // finally i convert every 2D filter to 1D vectors
   int n1 = cnn.convoLayers[0].numFilters;
   int n2 = cnn.convoLayers[0].inputDim_z;
   arma::mat filterMap(n1*n2, 25);
   int count = 0;
   for (int i = 0; i < cnn.convoLayers[0].numFilters; i++){
       for (int j = 0; j < cnn.convoLayers[0].inputDim_z; j++){
           filterMap.row(count++) = arma::vectorise((*cnn.convoLayers[0].filters)[i][j],1);
       }
   }
   
   filterMap.save("finalFilter.dat",arma::raw_ascii);
    
    
}

void workOnMNIST(){
    std::shared_ptr<arma::mat> DataX(new arma::mat);
    std::shared_ptr<arma::mat> DataY(new arma::mat);
    std::shared_ptr<arma::mat> trainDataX(new arma::mat);
    std::shared_ptr<arma::mat> trainDataY(new arma::mat);
    std::shared_ptr<arma::mat> testDataX(new arma::mat);
    std::shared_ptr<arma::mat> testDataY(new arma::mat);
    std::shared_ptr<arma::mat> ValidationDataX(new arma::mat);
    std::shared_ptr<arma::mat> ValidationDataY(new arma::mat);

    loadData_MNIST(DataX,DataY);

    int ntrain =2000;
    int ntest = 100;
//  now I split data into train, test, and validation
    trainDataX = std::make_shared<arma::mat>(DataX->rows(0,ntrain-1));
    trainDataY = std::make_shared<arma::mat>(DataY->rows(0,ntrain-1));
    testDataX = std::make_shared<arma::mat>(DataX->rows(ntrain,ntrain+ntest-1));
    testDataY = std::make_shared<arma::mat>(DataY->rows(ntrain,ntrain+ntest-1));
    TrainingPara trainingPara(1e-6,30, 1, 0.2);
    int fSize = 28;
    std::shared_ptr<arma::cube> trainDataX2D(new arma::cube(fSize,fSize,ntrain));
    std::shared_ptr<arma::cube> testDataX2D(new arma::cube(fSize,fSize,ntest));

    for (int i = 0 ; i < ntrain; i++) {
        for(int j = 0; j < fSize; j++) {
            for( int k = 0; k < fSize; k++) {
                (*trainDataX2D)(j,k,i) = trainDataX->at(i,fSize*j+k);
            }
        }
    }
    
    for (int i = 0 ; i < ntest; i++) {
        for(int j = 0; j < fSize; j++) {
            for( int k = 0; k < fSize; k++) {
                (*testDataX2D)(j,k,i) = testDataX->at(i,fSize*j+k);
            }
        }
    }

    DataX.reset();
    DataY.reset();
    
//    trainDataX2D = std::make_shared<arma::cube>(1,2,1,arma::fill::ones);
//    trainDataY = std::make_shared<arma::mat>(1,2,arma::fill::ones);
    
   CNN cnn(trainDataX2D, trainDataY, 1, std::move(trainingPara));
   cnn.train();
   cnn.test(testDataX2D, testDataY);
   // finally i convert every 2D filter to 1D vectors
   int n1 = cnn.convoLayers[0].numFilters;
   int n2 = cnn.convoLayers[0].inputDim_z;
   arma::mat filterMap(n1*n2, 9);
   int count = 0;
   for (int i = 0; i < cnn.convoLayers[0].numFilters; i++){
       for (int j = 0; j < cnn.convoLayers[0].inputDim_z; j++){
           filterMap.row(count++) = arma::vectorise((*cnn.convoLayers[0].filters)[i][j],1);
       }
   }
   
   filterMap.save("finalFilter.dat",arma::raw_ascii);
   

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
    int numSamples = 1000;
    X->set_size(numFiles*numSamples,featSize);
    Y->set_size(numFiles*numSamples,labelSize);
    Y->fill(0);


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
//        std::cout << x << std::endl;
                    (*X)(i+numFiles*j,k)=((unsigned char)x)/256.0;

                }
                (*Y)(i+numFiles*j,i) = 1;
//        count++;
            }

        } else {
            std::cout << "open file failure!" << std::endl;
        }
        std::cout << "dataloading finish!" <<std::endl;

    }

}
