#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <armadillo>
#include <vector>
#include "MultiLayerPerceptron.h"
#include "../Optimization/optimization.h"


using namespace NeuralNet;

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

    int ntrain =2000;
    int ntest = 1000;
//  now I split data into train, test, and validation
    trainDataX = std::make_shared<arma::mat>(DataX->cols(0,ntrain-1));
    trainDataY = std::make_shared<arma::mat>(DataY->cols(0,ntrain-1));
    testDataX = std::make_shared<arma::mat>(DataX->cols(ntrain,ntrain+ntest-1));
    testDataY = std::make_shared<arma::mat>(DataY->cols(ntrain,ntrain+ntest-1));


    int inputDim = trainDataX->n_cols;
    int outputDim = trainDataY->n_cols;
    trainDataX->save("trainingSamples.txt",arma::raw_ascii);
    TrainingPara_MLP trainingPara(1e-6,100, 10, 0.25);
    trainingPara.print();
    std::vector<int> dimensions = {784,100,10};
    MultiLayerPerceptron mlp(2, dimensions, trainDataX, trainDataY, trainingPara);
    bool LBFGS_flag = false;
    if (LBFGS_flag){
    MLPTrainer mlpTrainer(mlp);
    Optimization::LBFGS::LBFGS_param param(100,20, 50 , "result.txt");
    Optimization::LBFGS lbfgs_opt(mlpTrainer,param, Optimization::LBFGS::Wolfe);
    lbfgs_opt.minimize();
    } else{
    mlp.train();
    }
    mlp.test(testDataX,testDataY);


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
    X->set_size(featSize, numFiles*numSamples);
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

// for (int j = 0 ; j < numSamples ; j++){
//       for (int k =0 ; k <featSize; k ++){

//	           std::cout << x << std::endl;
//	   std::cout<<  (*X)(j,k) << " ";
//	   }
//	   }

        std::cout << "dataloading finish!" <<std::endl;

    }

}
