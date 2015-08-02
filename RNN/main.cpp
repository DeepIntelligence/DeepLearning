#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <armadillo>
#include <vector>
#include <math.h>

#include "RNN.h"

using namespace NeuralNet;

void workOnSequenceGeneration(std::shared_ptr<arma::mat> trainingY);
void testForward();
void trainRNN();
void testGrad();
void testDynamics();

int main(int argc, char *argv[]) {
//    testForward();
//    workOnSequenceGeneration();
   // testGrad();
//    trainRNN();
    testDynamics();
    return 0;
}

// use LSTM to approximate a dynamical system
void testDynamics(){
    
    std::shared_ptr<arma::mat> trainingX(new arma::mat(1,10));
    std::shared_ptr<arma::mat> trainingY(new arma::mat(1,10));
    
    // initialize 
    trainingX->zeros();
    trainingY->at(0) = 1.1;
    //trainingY->at(1) = 0.2;
    //trainingY->at(2) = 0.1;
    for (int i = 1; i < trainingY->n_elem; i++){
        //trainingY->at(i) = sin(trainingY->at(i-1)); // sine wave xt = sin(xt-1))
        // trainingY->at(i) = pow(trainingY->at(i-1),1); // xt = xt-1 ^ 2 
        //trainingY->at(i) = trainingY->at(i-1) + trainingY->at(i-2) - trainingY->at(i-3);
        trainingY->at(i) = trainingY->at(i-1)*trainingY->at(i-1);
//        trainingY->at(i) = sin(i);
        
    }
    
    // trainingY->ones(1, 100);
   
    int iterations = 5000;

    /* RNN constructor parameters passed as:
        RNN(int numHiddenLayers0, int hiddenLayerInputDim0,
        int hiddenLayerOutputDim0, int inputDim0, int outputDim0, 
        std::shared_ptr<arma::mat> trainingX0, std::shared_ptr<arma::mat> trainingY0)
     */
    RNN rnn(4, 8, 8, 1, 1, trainingX, trainingY);
    // train the LSTM model by iterations
    for (int iter = 0; iter < iterations; iter++) {
        rnn.train();
    }

    trainingY->save("trainingY.dat", arma::raw_ascii);
    for (int k = 0; k < trainingY->n_elem; k++) {
        (rnn.getOutputLayer()->outputMem[k])->print();
    }
    trainingY->print();
}

// test the gradients by numerical gradients checking
void testGrad() {
    
    std::shared_ptr<arma::mat> trainingX(new arma::mat);
    std::shared_ptr<arma::mat> trainingY(new arma::mat);
    
    trainingX->randn(1, 10);
     trainingY->ones(1, 10);
    /* RNN constructor parameters passed as:
        RNN(int numHiddenLayers0, int hiddenLayerInputDim0,
        int hiddenLayerOutputDim0, int inputDim0, int outputDim0, 
        std::shared_ptr<arma::mat> trainingX0, std::shared_ptr<arma::mat> trainingY0)
     */
    RNN rnn(2, 2, 2, 1, 1, trainingX, trainingY);
    // before applying the LSTM backprop model, generate numerical gradients by just forward pass.
    rnn.calNumericGrad();
    
    rnn.forward();
    rnn.backward();

    
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
    trainingY->ones(1, 10);
//RNN::RNN_LSTM(int numHiddenLayers0, int hiddenLayerInputDim0,
//        int hiddenLayerOutputDim0, int inputDim0, int outputDim0, 
//        std::shared_ptr<arma::mat> trainingX0, std::shared_ptr<arma::mat> trainingY0)
    
    RNN rnn(1, 2, 2, 1, 1, trainingX, trainingY);
    rnn.forward();
    rnn.backward();

}

void trainRNN(){
    /*std::shared_ptr<arma::mat> trainingX(new arma::mat(1,10));
    
    trainingX->randn(1, 10);
    std::shared_ptr<arma::mat> trainingY(new arma::mat());
    trainingY->ones(3, 10);
     */
    
    std::shared_ptr<arma::mat> trainingX(new arma::mat());

    std::shared_ptr<arma::mat> trainingY(new arma::mat());
//    trainingY->load("testdata.dat");
//    *trainingY = arma::trans(*trainingY);
//    trainingX->zeros(trainingY->n_cols,1);
    

    
    trainingX->zeros(1, 10);
    trainingY->zeros(1,10);
    for (int i = 0; i <10; i++)
        trainingY->at(i) = i;
    trainingY->transform([](double val){return sin(val);});
    
    int iterations = 50000;
    
    /* RNN constructor parameters passed as:
        RNN(int numHiddenLayers0, int hiddenLayerInputDim0,
        int hiddenLayerOutputDim0, int inputDim0, int outputDim0, 
        std::shared_ptr<arma::mat> trainingX0, std::shared_ptr<arma::mat> trainingY0)
     */
    RNN rnn(2, 5, 5, 1, 1, trainingX, trainingY);
    // train the LSTM model by iterations
    for (int iter = 0; iter < iterations;iter++){
        rnn.train();
    }
    
    trainingY->save("trainingY.dat",arma::raw_ascii);
    for (int k = 0; k < 10; k++){
        (rnn.getOutputLayer()->outputMem[k])->print();
    }
}

