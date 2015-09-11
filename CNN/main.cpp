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

load_cifar10(X, Y, filename,1,600);
// first try to plot some image
//for (int i = 0; i < 100; i++) {
//    arma::cube temp = X->slices(i*3,i*3+2);
//    temp = temp * 255;
//    temp.reshape(1,32*32);
//    char tag[10];
//    sprintf(tag,"%d",i);
//    temp.save("image" + (std::string)tag, arma::ppm_binary);
//}

    int ntrain = 500;
    int ntest = 100;
    std::shared_ptr<arma::mat> trainDataY(new arma::mat);
    std::shared_ptr<arma::mat> testDataY(new arma::mat);
    int fSize = 32;
    std::shared_ptr<arma::cube> trainDataX2D(new arma::cube(fSize,fSize,ntrain));
    std::shared_ptr<arma::cube> testDataX2D(new arma::cube(fSize,fSize,ntest));    
    

    int nChannel = 3;
    trainDataY = std::make_shared<arma::mat>(Y->cols(0,ntrain-1));
    testDataY = std::make_shared<arma::mat>(Y->cols(ntrain,ntrain+ntest-1));
    trainDataX2D = std::make_shared<arma::cube>(X->slices(0,ntrain*nChannel-1));
    testDataX2D = std::make_shared<arma::cube>(X->slices(ntrain*nChannel,ntrain*nChannel+ntest*nChannel-1));


TrainingPara trainingPara(1e-6, 100, 10, 0.01);



   CNN cnn(trainDataX2D, trainDataY,3, std::move(trainingPara));
      bool LBFGS_flag = true;
    if (LBFGS_flag){
    CNNTrainer cnnTrainer(cnn);
//    cnnTrainer.gradientChecking();
    Optimization::LBFGS::LBFGS_param param(100,20,20,"lbfgs_weight.dat");
    Optimization::LBFGS lbfgs_opt(cnnTrainer,param, Optimization::LBFGS::Wolfe);
    lbfgs_opt.minimize();
    
    } else {
   cnn.train();
    }
    
    cnn.test(testDataX2D, testDataY);
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

    int ntrain = 500;
    int ntest = 100;
//  now I split data into train, test, and validation
    trainDataX = std::make_shared<arma::mat>(DataX->cols(0,ntrain-1));
    trainDataY = std::make_shared<arma::mat>(DataY->cols(0,ntrain-1));
    testDataX = std::make_shared<arma::mat>(DataX->cols(ntrain,ntrain+ntest-1));
    testDataY = std::make_shared<arma::mat>(DataY->cols(ntrain,ntrain+ntest-1));

    int fSize = 28;
    std::shared_ptr<arma::cube> trainDataX2D(new arma::cube(fSize,fSize,ntrain));
    std::shared_ptr<arma::cube> testDataX2D(new arma::cube(fSize,fSize,ntest));

    for (int i = 0 ; i < ntrain; i++) {
        for(int j = 0; j < fSize; j++) {
            for( int k = 0; k < fSize; k++) {
                (*trainDataX2D)(k,j,i) = trainDataX->at(fSize*j+k, i);
            }
        }
    }
    
    for (int i = 0 ; i < ntest; i++) {
        for(int j = 0; j < fSize; j++) {
            for( int k = 0; k < fSize; k++) {
                (*testDataX2D)(k,j,i) = testDataX->at(fSize*j+k, i);
            }
        }
    }

    DataX.reset();
    DataY.reset();
    
//    trainDataX2D = std::make_shared<arma::cube>(1,2,1,arma::fill::ones);
//    trainDataY = std::make_shared<arma::mat>(1,2,arma::fill::ones);
     TrainingPara trainingPara(1e-6,20, 50, 0.2, 10, true);  
   CNN cnn(trainDataX2D, trainDataY, 1, std::move(trainingPara));
   
   bool LBFGS_flag = false;
    if (LBFGS_flag){
    CNNTrainer cnnTrainer(cnn);
//    cnnTrainer.gradientChecking();
    Optimization::LBFGS::LBFGS_param param(100,20,50,"lbfgs_weight.dat");
    Optimization::LBFGS lbfgs_opt(cnnTrainer,param, Optimization::LBFGS::Wolfe);
    lbfgs_opt.minimize();
    
    } else {
    cnn.train();
    }
   cnn.test(testDataX2D, testDataY);


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
   X->set_size(featSize,numFiles*numSamples);
    Y->set_size(labelSize, numFiles*numSamples);
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
                    (*X)(k, i+numFiles*j)=((unsigned char)x)/256.0;

                }
                (*Y)(i, i+numFiles*j) = 1;
//        count++;
            }

        } else {
            std::cout << "open file failure!" << std::endl;
        }
        std::cout << "dataloading finish!" <<std::endl;

    }

}
