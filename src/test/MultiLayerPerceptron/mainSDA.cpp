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



