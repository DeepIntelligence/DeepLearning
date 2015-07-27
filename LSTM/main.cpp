#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <armadillo>
#include <vector>
#include <math.h>

#include "LSTMLayer.h"

using namespace NeuralNet;

void workOnSequenceGeneration(std::shared_ptr<arma::mat> trainingY);
void testForward();
void trainRNN_LSTM();
void testGrad();

void genSimData(); // generate simulation data
double f_x(double t);

int main(int argc, char *argv[]) {
   // testForward();
//    workOnSequenceGeneration();
    testGrad();
    return 0;
}

// test the gradients by numerical gradients checking
void testGrad() {
    
    std::shared_ptr<arma::mat> trainingX(new arma::mat(1,10));
    trainingX->randn(1, 10);
    std::shared_ptr<arma::mat> trainingY(new arma::mat());
    trainingY->ones(3, 10);

    /* RNN_LSTM constructor parameters passed as:
        RNN_LSTM(int numHiddenLayers0, int hiddenLayerInputDim0,
        int hiddenLayerOutputDim0, int inputDim0, int outputDim0, 
        std::shared_ptr<arma::mat> trainingX0, std::shared_ptr<arma::mat> trainingY0)
     */
    RNN_LSTM rnn(1, 2, 2, 2, 1, trainingX, trainingY);
    // before applying the LSTM backprop model, generate numerical gradients by just forward pass.
    rnn.calNumericGrad();
    // train the LSTM model by one iteration to generate gradient from the model
    rnn.train();
    
}

void workOnSequenceGeneration(std::shared_ptr<arma::mat> trainingY){
    // std::shared_ptr<arma::mat> trainingY(new arma::mat);
    // junhyukoh's data with only y labeled, no input x
    trainingY->load("testdata.dat",arma::raw_ascii);
    trainingY->print();
}


void testForward(){
    
    std::shared_ptr<arma::mat> trainingX(new arma::mat());
    trainingX->randn(1, 10);
    std::shared_ptr<arma::mat> trainingY(new arma::mat());
    trainingY->ones(3, 10);
//RNN_LSTM::RNN_LSTM(int numHiddenLayers0, int hiddenLayerInputDim0,
//        int hiddenLayerOutputDim0, int inputDim0, int outputDim0, 
//        std::shared_ptr<arma::mat> trainingX0, std::shared_ptr<arma::mat> trainingY0)
    
    RNN_LSTM rnn(1, 2, 2, 1, 1, trainingX, trainingY);
    rnn.forward();
    rnn.backward();

}

void trainRNN_LSTM(){
    /*std::shared_ptr<arma::mat> trainingX(new arma::mat(1,10));
    
    trainingX->randn(1, 10);
    std::shared_ptr<arma::mat> trainingY(new arma::mat());
    trainingY->ones(3, 10);
     */
    
    std::shared_ptr<arma::mat> trainingY(new arma::mat());
    workOnSequenceGeneration(trainingY);
    std::shared_ptr<arma::mat> trainingX(new arma::mat(1, trainingY->n_cols));
    for(int t=0; t<trainingY->n_cols;t++){
       (*trainingX)(1,t) = 0;
    }
    
    
    int iterations = 100;
    
    /* RNN_LSTM constructor parameters passed as:
        RNN_LSTM(int numHiddenLayers0, int hiddenLayerInputDim0,
        int hiddenLayerOutputDim0, int inputDim0, int outputDim0, 
        std::shared_ptr<arma::mat> trainingX0, std::shared_ptr<arma::mat> trainingY0)
     */
    RNN_LSTM rnn(3, 2, 2, 1, 1, trainingX, trainingY);
    // train the LSTM model by iterations
    for (int iter = 0; iter<=iterations;iter++){
        rnn.train();
    }
    
}

/*void genSimData(std::shared_ptr<arma::mat> trainingX){

    int TotalLength = 10;
    double mean = 0;
    double max_abs = 0;
    for (int i = 0; i < TotalLength; ++i) {
        double val = f_x(i * 0.01);
        max_abs = max(max_abs, abs(val));
    }
    for (int i = 0; i < TotalLength; ++i) {
        mean += f_x(i * 0.01) / max_abs;
    }
    mean /= TotalLength;
    for (int i = 0; i < TotalLength; ++i) {
        trainingX[i] = f_x(i * 0.01) / max_abs - mean;
    }
    
    
}

double f_x(double t) {
    
    return 0.5 * sin(2 * t) - 0.05 * cos(17 * t + 0.8)
            + 0.05 * sin(25 * t + 10) - 0.02 * cos(45 * t + 0.3);
    
}*/