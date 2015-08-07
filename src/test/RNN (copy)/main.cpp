#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <armadillo>
#include <vector>
#include <math.h>
#include <random>

#include "RNN.h"

using namespace NeuralNet;

void workOnSequenceGeneration(std::shared_ptr<arma::mat> trainingY);
void testForward();
void trainRNN();
void testGrad();
void testDynamics();
void aLittleTimerGenerator(std::shared_ptr<arma::mat> trainingX,  
        std::shared_ptr<arma::mat> trainingY);

int main(int argc, char *argv[]) {
//    testForward();
//    workOnSequenceGeneration();
   // testGrad();
//    trainRNN();
    testDynamics();
    return 0;
}

void aLittleTimerGenerator(std::shared_ptr<arma::mat> trainingX,  
        std::shared_ptr<arma::mat> trainingY){
    
    int T = trainingY->n_elem;
    
    std::random_device device;
    std::mt19937 gen(device());
    std::bernoulli_distribution distribution(0.1);
    std::uniform_real_distribution<> dis(0, 1);
    
    arma::mat input(2, T);
    arma::mat output(1, T);
    
    
    for(int i=0;i<T;i++){
        if (distribution(gen)){
            input(0,i) = 1;
        }
        else{
            input(0,i) = 0;
        }
    }
    
    // generate input u2(t)
    input(1,0) = (int)((dis(gen) + 0.1)*10);
    for(int i=1;i<T;i++){
        if( input(0,i) == 1) {
            input(1,i) = (int)((dis(gen) + 0.1)*10);
        } else {
            input(1,i) = input(1,i-1); // if input1 = 0,keep the input2 the same as previously
        }
    }
    
    std::cout<<"input_1"<<std::endl;
    input.row(0).print(); 
    std::cout<<std::endl;
    
    
    // generate output
    int t = 0;
    while(t<T){
        
        if(input(0,t)==1){
            
            output(arma::span(0,0),arma::span(t,T-1)).zeros();
           
            //std::cout<<"input(1,t)"<<input(1,t)<<std::endl;
            for (int i=t; i<(t+input(1,t));i++){
                //std::cout<<i<<" ";
                if(i<T){
                    output(0,i) = 0.5;
                }
                else{
                    break;
                }
            } 
           
        }
        else{
            if(output(0,t) == 0.5){
                output(0,t) = 0.5;
            }
            else{
                output(0,t) = 0;
            }
            
        }
        //output(0,arma::span(t,t)).print();
        t = t + 1;
    }
    
    input.row(1) = input.row(1) / 10;
    
    std::cout<<"input_2"<<std::endl;
    input.row(1).print();
    std::cout<<std::endl;
    //input.row(1).print();
    std::cout<<std::endl;      
    std::cout<<t<<std::endl;
    std::cout<<"output"<<std::endl;
    output.print();
    
    
    *trainingX = input;
    *trainingY = output;
}

// use LSTM to approximate a dynamical system
void testDynamics(char* filename, NeuralNetParameter message){
    
    //NeuralNetParameter message; 
    // ReadProtoFromTextFile(filename, &message);	
    
    //int seriesLength = message.rnnstruct().timeserieslength();
    int seriesLength = 50;
    std::cout << seriesLength << std::endl;
    
    std::shared_ptr<arma::mat> trainingX(new arma::mat(1,seriesLength));
    std::shared_ptr<arma::mat> trainingY(new arma::mat(1,seriesLength));
    /*
    // initialize 
    trainingX->zeros();
    trainingY->at(0) = 0.9999;
    //trainingY->at(1) = 0.2;
    //trainingY->at(2) = 0.1;
    for (int i = 1; i < trainingY->n_elem; i++){
        //trainingY->at(i) = sin(trainingY->at(i-1)); // sine wave xt = sin(xt-1))
        // trainingY->at(i) = pow(trainingY->at(i-1),1); // xt = xt-1 ^ 2 
        //trainingY->at(i) = trainingY->at(i-1) + trainingY->at(i-2) - trainingY->at(i-3);
        trainingY->at(i) = trainingY->at(i-1)*trainingY->at(i-1);
//        trainingY->at(i) = sin(i);
        
    }
    */
    aLittleTimerGenerator(trainingX, trainingY);
    
    int iterations = message.neuralnettrainingparameter().nepoch();
    
    /* RNN constructor parameters passed as:
        RNN(int numHiddenLayers0, int hiddenLayerInputDim0,
        int hiddenLayerOutputDim0, int inputDim0, int outputDim0, double learningRate,
        std::shared_ptr<arma::mat> trainingX0, std::shared_ptr<arma::mat> trainingY0)
     */
    RNN rnn(4, 8, 8, 1, 1, trainingX, trainingY);
    // train the LSTM model by iterations
    for (int iter = 0; iter < iterations; iter++) {
        rnn.train();
    }
    
    // train the RNN model by iterations
    for (int iter = 0; iter < iterations; iter++) {
        rnn.train();
    }

    trainingY->save("trainingY.dat", arma::raw_ascii);
    for (int k = 0; k < trainingY->n_elem; k++) {
        (rnn.getOutputLayer()->outputMem[k])->print();
    }
    trainingY->print();
    
}
/*
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
} */


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

