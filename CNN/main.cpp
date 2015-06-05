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

int main(int argc, char *argv[]) {
    std::shared_ptr<arma::mat> DataX(new arma::mat);
    std::shared_ptr<arma::mat> DataY(new arma::mat);
    std::shared_ptr<arma::mat> trainDataX(new arma::mat);
    std::shared_ptr<arma::mat> trainDataY(new arma::mat);
    std::shared_ptr<arma::mat> testDataX(new arma::mat);
    std::shared_ptr<arma::mat> testDataY(new arma::mat);
    std::shared_ptr<arma::mat> ValidationDataX(new arma::mat);
    std::shared_ptr<arma::mat> ValidationDataY(new arma::mat);

    loadData_MNIST(DataX,DataY);

    int ntrain = 500;
    int ntest = 100;
//  now I split data into train, test, and validation
    trainDataX = std::make_shared<arma::mat>(DataX->rows(0,ntrain-1));
    trainDataY = std::make_shared<arma::mat>(DataY->rows(0,ntrain-1));
    testDataX = std::make_shared<arma::mat>(DataX->rows(ntrain,ntrain+ntest-1));
    testDataY = std::make_shared<arma::mat>(DataY->rows(ntrain,ntrain+ntest-1));
    TrainingPara trainingPara(1e-6,50, 1, 0.25);
    std::shared_ptr<arma::cube> trainDataX2D(new arma::cube(28,28,ntrain));
//    MatArray<double>::Mat1DArray_ptr trainDataX2D2 = MatArray<double>::build(ntrain);

    for (int i = 0 ; i < ntrain; i++) {
//        (*trainDataX2D2)[i].set_size(28,28);
        for(int j = 0; j < 28; j++) {
            for( int k = 0; k < 28; k++) {
                (*trainDataX2D)(j,k,i) = trainDataX->at(i,28*j+k);
//                (*trainDataX2D2)[i](j,k) = trainDataX->at(i,28*j+k);
            }
        }
//        (*trainDataX2D2)[i].print();
    }

    trainDataX2D->save("cube.dat",arma::raw_ascii);
    DataX.reset();
    DataY.reset();
    
   CNN cnn(trainDataX2D, trainDataY, 1, std::move(trainingPara));
   cnn.train();
   // finally i convert every 2D filter to 1D vectors
   int n1 = cnn.convoLayers[0].numFilters;
   int n2 = cnn.convoLayers[0].inputDim_z;
   arma::mat filterMap(n1*n2, 81);
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

// for (int j = 0 ; j < numSamples ; j++){
//       for (int k =0 ; k <featSize; k ++){

//	           std::cout << x << std::endl;
//	   std::cout<<  (*X)(j,k) << " ";
//	   }
//	   }

        std::cout << "dataloading finish!" <<std::endl;

    }

}
