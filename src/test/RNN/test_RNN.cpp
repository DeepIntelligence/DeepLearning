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
#include "common.h"
#include "../Trainer/Trainer.h"

using namespace NeuralNet;
using namespace DeepLearning;
void workOnSequenceGeneration(std::shared_ptr<arma::mat> trainingY);
void testForward();
void trainRNN();
void testGrad();
void testDynamicswithTrainer(char* filename);
void testDynamics();

int main(int argc, char *argv[]) {
//    testForward();
//    workOnSequenceGeneration();
   // testGrad();
//    trainRNN();
    testDynamicswithTrainer(argv[1]);
    return 0;
}

void testDynamicswithTrainer(char* filename){
    
    std::shared_ptr<arma::mat> trainingX(new arma::mat(1,10));
    std::shared_ptr<arma::mat> trainingY(new arma::mat(1,10));
    
    // initialize 
    trainingX->zeros();
    trainingY->at(0) = 1.001;
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
    NeuralNetParameter message; 
    ReadProtoFromTextFile(filename, &message);
    
    std::shared_ptr<Net> rnn(new RNN(message));
    std::shared_ptr<Trainer> trainer(TrainerBuilder::GetTrainer(rnn,message));
    traistd::shared_ptr<arma::mat> trainingX, trainingY;
    ner->setTrainingSamples(trainingX, trainingY);
    trainer->train();
    
    trainingY->save("trainingY.dat", arma::raw_ascii);
    (rnn->netOutput())->print();
    trainingY->print();
}

void testDynamics(){
    
    std::shared_ptr<arma::mat> trainingX(new arma::mat(1,10));
    std::shared_ptr<arma::mat> trainingY(new arma::mat(1,10));
    
    // initialize 
    trainingX->zeros();
    trainingY->at(0) = 0.99;
    //trainingY->at(1) = 0.2;
    //trainingY->at(2) = 0.1;
    for (int i = 1; i < trainingY->n_elem; i++){
        //trainingY->at(i) = sin(trainingY->at(i-1)); // sine wave xt = sin(xt-1))
        // trainingY->at(i) = pow(trainingY->at(i-1),1); // xt = xt-1 ^ 2 
        //trainingY->at(i) = trainingY->at(i-1) + trainingY->at(i-2) - trainingY->at(i-3);
        trainingY->at(i) = trainingY->at(i-1)*trainingY->at(i-1);
//        trainingY->at(i) = sin(i);
        
    }
    
    int iterations = 20000;

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
}

