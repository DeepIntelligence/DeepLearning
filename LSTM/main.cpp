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
    //testGrad();
    trainRNN_LSTM();
    return 0;
}

// test the gradients by numerical gradients checking

void testGrad() {

    std::shared_ptr<arma::mat> trainingX(new arma::mat(1, 10));
    trainingX->randu(1, 10);
    std::shared_ptr<arma::mat> trainingY(new arma::mat());
    trainingY->ones(1, 10);

    /* RNN_LSTM constructor parameters passed as:
        RNN_LSTM(int numHiddenLayers0, int hiddenLayerInputDim0,
        int hiddenLayerOutputDim0, int inputDim0, int outputDim0, 
        std::shared_ptr<arma::mat> trainingX0, std::shared_ptr<arma::mat> trainingY0)
     */
    RNN_LSTM rnn(3, 2, 2, 1, 1, trainingX, trainingY);
    // before applying the LSTM backprop model, generate numerical gradients by just forward pass.
    rnn.calNumericGrad();
    // train the LSTM model by one iteration to generate gradient from the model
    rnn.train();

}

void workOnSequenceGeneration(std::shared_ptr<arma::mat> trainingY) {
    // std::shared_ptr<arma::mat> trainingY(new arma::mat);
    // junhyukoh's data with only y labeled, no input x
    trainingY->load("testdata.dat", arma::raw_ascii);
    trainingY->print();
}

void testForward() {

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

void trainRNN_LSTM() {

    std::shared_ptr<arma::mat> trainingX(new arma::mat());

    std::shared_ptr<arma::mat> trainingY(new arma::mat());
    trainingX->zeros(1, 10);
    trainingY->zeros(1, 10);
    for (int i = 0; i < 10; i++)
        trainingY->at(i) = i;
    trainingY->transform([](double val) {
        return sin(val);
    });

    int iterations = 50000;

    /* RNN constructor parameters passed as:
        RNN(int numHiddenLayers0, int hiddenLayerInputDim0,
        int hiddenLayerOutputDim0, int inputDim0, int outputDim0, 
        std::shared_ptr<arma::mat> trainingX0, std::shared_ptr<arma::mat> trainingY0)
     */
    RNN_LSTM lstm(3, 5, 5, 1, 1, trainingX, trainingY);
    // train the LSTM model by iterations
    for (int iter = 0; iter < iterations; iter++) {
        lstm.train();
    }

    trainingY->save("trainingY.dat", arma::raw_ascii);
    for (int k = 0; k < 10; k++) {
        (lstm.getOutputLayer()->outputMem[k])->print();
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