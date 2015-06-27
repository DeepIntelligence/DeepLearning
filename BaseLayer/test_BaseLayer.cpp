#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <armadillo>
#include "BaseLayer.h"
#include "gtest/gtest.h"
using namespace NeuralNet;

void loadData_MNIST(std::shared_ptr<arma::mat> X,
                    std::shared_ptr<arma::mat> Y,
                    std::string filename_base);

TEST(BaseLayerTest, fillBernoulli){

    BaseLayer layer(100,10,BaseLayer::sigmoid,true,0.5);
    EXPECT_EQ(layer.dropOutRate,0.5);
//    EXPECT_TRUE(layer.dropOutFlag);
    layer.B->print();
    layer.fill_Bernoulli(layer.B->memptr(),layer.B_size);
    layer.B->print();
    
}
 


int main(int argc, char *argv[]) {
    std::shared_ptr<arma::mat> trainDataX(new arma::mat);
    std::shared_ptr<arma::mat> trainDataY(new arma::mat);
    loadData_MNIST(trainDataX,trainDataY, "../MNIST/data");

    BaseLayer baseLayer(784,10,BaseLayer::softmax);
    baseLayer.inputX = trainDataX;
    baseLayer.activateUp(baseLayer.inputX);

//  baseLayer.inputX->print();

    baseLayer.save();

//    trainDataX->save("X.dat",arma::raw_ascii);
//    baseLayer.outputY->save("PredY.dat",arma::raw_ascii);
    
    
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}


void loadData_MNIST(std::shared_ptr<arma::mat> X,
                    std::shared_ptr<arma::mat> Y,
                    std::string filename_base) {

    std::string filename;
    char tag[50];
    char x;
    int count;
    int numFiles = 10;
    int featSize = 28*28;
    int labelSize = 10;
    int numSamples = 10;
    X->set_size(featSize,numFiles*numSamples);
    Y->set_size(labelSize,numFiles*numSamples);
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
//        std::cout << x << std::endl;
                    (*X)(k, j+i*numSamples)=(unsigned char)x;
                }
                (*Y)(i,j+i*numSamples) = 1;
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
