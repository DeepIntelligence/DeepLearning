#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <armadillo>
#include "RBM.h"
#include "ProgramArgs.h"

using namespace NeuralNet;

void loadData_MNIST(std::shared_ptr<arma::mat> X,
                    std::shared_ptr<arma::mat> Y, const std::string);

int main(int argc, char *argv[]) {
    std::shared_ptr<arma::mat> DataX(new arma::mat);
    std::shared_ptr<arma::mat> DataY(new arma::mat);
    std::shared_ptr<arma::mat> trainDataX(new arma::mat);
    std::shared_ptr<arma::mat> trainDataY(new arma::mat);
    std::shared_ptr<arma::mat> testDataX(new arma::mat);
    std::shared_ptr<arma::mat> testDataY(new arma::mat);
    std::shared_ptr<arma::mat> ValidationDataX(new arma::mat);
    std::shared_ptr<arma::mat> ValidationDataY(new arma::mat);
    
    ProgramArgs progArgs(argc, argv);
    
    loadData_MNIST(DataX,DataY, progArgs.dataPath);

    int ntrain = progArgs.ntrain;
    int ntest = progArgs.ntest;  
    int hiddenDim = progArgs.hiddenDim;
    int inputDim = progArgs.inputDim;
    
    RBM::PreTrainPara trainingPara(progArgs.eps, progArgs.nEpoch, progArgs.miniBatchSize,
            progArgs.learningRate, progArgs.momentum, progArgs.saveFrequency, progArgs.learningRateDecay, 
            progArgs.dropOutFlag, progArgs.dropOutRate);    
//  now I split data into train, test, and validation
    trainDataX = std::make_shared<arma::mat>(DataX->cols(0,ntrain-1));
    trainDataY = std::make_shared<arma::mat>(DataY->cols(0,ntrain-1));
    testDataX = std::make_shared<arma::mat>(DataX->cols(ntrain,ntrain+ntest-1));
    testDataY = std::make_shared<arma::mat>(DataY->cols(ntrain,ntrain+ntest-1));

    DataX.reset();
    DataY.reset();



    std::cout << trainDataX->n_cols << std::endl;
 
    trainingPara.print();

    bool trainFlag = true;
    bool testFlag = true;
    
    std::string filename = "pretrain_final";
    std::shared_ptr<arma::umat> trainDataXBin(new arma::umat(trainDataX->n_rows,trainDataX->n_cols));
    *trainDataXBin = (*trainDataX) > 0.5;
    RBM rbm(inputDim, hiddenDim, trainDataXBin, trainingPara);

    if (trainFlag) {
        rbm.train();
        rbm.saveTrainResult(filename);
    }

    if (testFlag) {
        if (!trainFlag) rbm.loadTrainResult(filename);
        testDataX->save("testSample.dat",arma::raw_ascii);
        rbm.TestViaReconstruct(testDataX);
    }
}



void loadData_MNIST(std::shared_ptr<arma::mat> X,
                    std::shared_ptr<arma::mat> Y,const std::string filepath) {

    std::string filename_base(filepath);
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

