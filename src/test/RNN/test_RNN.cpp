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
#include "common.h"
#include "../Trainer/Trainer.h"

using namespace NeuralNet;
using namespace DeepLearning;
void testGrad(char* filename);
void testLittleTimer(char* filename);
void testSimpleDynamics(char* filename);
void testIntermediateDynamics(char* filename);
void testForward(char* filename);
void testRLData(char* filename);
void aLittleTimerGenerator(std::shared_ptr<arma::mat> trainingX,  
        std::shared_ptr<arma::mat> trainingY);

std::random_device device;
std::mt19937 gen(device());
std::bernoulli_distribution distribution(0.1);
std::uniform_real_distribution<> dis(0, 1);

int main(int argc, char *argv[]) {
//    testGrad(argv[1]);
//    testLittleTimer(argv[1]);
    testSimpleDynamics(argv[1]);
//    testIntermediateDynamics(argv[1]);
//        testRLData(argv[1]);
//    testForward(argv[1]);
    return 0;
}

void testForward(char* filename){
    double T = 10;
    std::shared_ptr<arma::mat> trainingX(new arma::mat(1,T));
    std::shared_ptr<arma::mat> trainingY(new arma::mat(1,T));
    arma::arma_rng::set_seed_random();
    trainingX->randn();
    trainingY->at(0) = 0;
    for (int i = 1; i < T; i++)
        trainingY->at(i) = sin(0.01*i) + 0.1* std::abs(trainingY->at(i));
    for (int i = 1; i < T; i++)
        trainingY->at(i) = 1.0*i / T;


    NeuralNetParameter message; 
    ReadProtoFromTextFile(filename, &message);
    
    std::shared_ptr<RNN> rnnptr(new RNN(message));
    
    rnnptr->setTrainingSamples(trainingX, nullptr);
    rnnptr->forward();
    (rnnptr->netOutput())->print();
    std::vector<RecurrLayer> vec = rnnptr->getHiddenLayers();
    for (int i = 0; i < vec[0].outputMem.size(); i++){
        vec[0].outputMem[i]->print("start");
        std::cout << vec[0].outputMem[i].get() << std::endl;
    
    }
}
void testRLData(char* filename){
   std::shared_ptr<arma::mat> trainingX(new arma::mat);
    std::shared_ptr<arma::mat> trainingY(new arma::mat);
    std::vector<std::shared_ptr<arma::mat>> X, Y;
    trainingX->load("X.dat");
    trainingY->load("Y.dat");
    X.push_back(trainingX);
    Y.push_back(trainingY);
    NeuralNetParameter message; 
    ReadProtoFromTextFile(filename, &message);
    
    std::shared_ptr<Net> rnn(new RNN(message));
    std::shared_ptr<Trainer> trainer(TrainerBuilder::GetTrainer(rnn,message));

    trainer->setTrainingSamples(trainingX, trainingY);
    trainer->train();
    
    trainingY->print("trainingY");
    (rnn->netOutput())->print("output");
}

void testIntermediateDynamics(char* filename){

        
    double T = 100;
    std::shared_ptr<arma::mat> trainingX(new arma::mat(1,T));
    std::shared_ptr<arma::mat> trainingY(new arma::mat(1,T));
    arma::arma_rng::set_seed_random();
    trainingX->zeros();
    trainingY->at(0) = 0;
    for (int i = 1; i < T; i++)
        trainingY->at(i) = sin(0.01*i) + 0.1* std::abs(trainingY->at(i));
    for (int i = 1; i < T; i++){
        trainingX->at(i) = 1.0*i / T;
        trainingY->at(i) = (1 - trainingX->at(i)) + 0.2*trainingY->at(i-1);
        if (i >= 2) {
        trainingY->at(i) = (1 - trainingX->at(i)) + 0.2*trainingY->at(i-1) - 0.3*trainingY->at(i-2);
        }
    }
    NeuralNetParameter message; 
    ReadProtoFromTextFile(filename, &message);
    
    std::shared_ptr<RNN> rnnptr(new RNN(message));
    std::shared_ptr<Net> rnn(rnnptr);
    

    rnn->setTrainingSamples(trainingX, nullptr);
    rnn->forward();
    rnn->netOutputAtTime(0);
    std::shared_ptr<Trainer> trainer(TrainerBuilder::GetTrainer(rnn,message));

    trainer->setTrainingSamples(trainingX, trainingY);
    trainer->train();
    
    trainingY->save("trainingY.dat", arma::raw_ascii);
    (rnn->netOutput())->print();
    std::cout<<std::endl;
    trainingY->print();
}

void testSimpleDynamics(char* filename){
    
    double T = 100;
    std::shared_ptr<arma::mat> trainingX(new arma::mat(1,T));
    std::shared_ptr<arma::mat> trainingY(new arma::mat(1,T));
    arma::arma_rng::set_seed_random();
    trainingX->zeros();
    trainingY->at(0) = 0;
    for (int i = 1; i < T; i++)
        trainingY->at(i) = sin(0.01*i) + 0.1* std::abs(trainingY->at(i));
    for (int i = 1; i < T; i++){
        trainingX->at(i) = 1.0*i / T;
        trainingY->at(i) = (1 - trainingX->at(i));
    }
    NeuralNetParameter message; 
    ReadProtoFromTextFile(filename, &message);
    
//    std::shared_ptr<RNN> rnnptr(new RNN(message));
    std::shared_ptr<Net> rnn(new RNN(message));
    

    rnn->setTrainingSamples(trainingX, nullptr);
    rnn->forward();
    rnn->netOutputAtTime(0);
    std::shared_ptr<Trainer> trainer(TrainerBuilder::GetTrainer(rnn,message));

    trainer->setTrainingSamples(trainingX, trainingY);
    trainer->train();
    
    trainingY->save("trainingY.dat", arma::raw_ascii);
    (rnn->netOutput())->print();
    std::cout<<std::endl;
    trainingY->print();
#if 0    
    std::vector<std::shared_ptr<arma::mat>> gradVec;
    rnn->calGradient();
    gradVec = rnn->netGradients();
    
    for (int i = 0; i < gradVec.size(); i++){
        std::cout << i << std::endl;
        gradVec[i]->print();
    }
    
    rnn->setTrainingSamples(trainingX, nullptr);
    rnn->forward();
    (rnn->netOutput())->print();
    rnn->save("RNN");
    std::vector<MultiAddLayer> vec = rnnptr->getHiddenLayers();
    for (int i = 0; i < vec[0].outputMem.size(); i++){
        vec[0].outputMem[i]->print("start");
        std::cout << vec[0].outputMem[i].get() << std::endl;
    
    }
#endif    

}

void testLittleTimer(char* filename){
    
    double T = 100;
    std::shared_ptr<arma::mat> trainingX(new arma::mat(1,T));
    std::shared_ptr<arma::mat> trainingY(new arma::mat(1,T));
    std::shared_ptr<arma::mat> testingX(new arma::mat(1,T));
    std::shared_ptr<arma::mat> testingY(new arma::mat(1,T));
    
    aLittleTimerGenerator(trainingX, trainingY);
    aLittleTimerGenerator(testingX, testingY);
    NeuralNetParameter message; 
    ReadProtoFromTextFile(filename, &message);
    
    std::shared_ptr<Net> rnn(new RNN(message));
    
    std::shared_ptr<Trainer> trainer(TrainerBuilder::GetTrainer(rnn,message));

    trainer->setTrainingSamples(trainingX, trainingY);
    trainer->train();
    
    trainingY->save("trainingY.dat", arma::raw_ascii);
    (rnn->netOutput())->print();
    std::cout<<std::endl;
    trainingY->print();
    
    rnn->setTrainingSamples(testingX, testingY);
    rnn->forward();
    (rnn->netOutput())->print("output for testing");
    testingY->print("testing Y");
    

}

void aLittleTimerGenerator(std::shared_ptr<arma::mat> trainingX,  
    std::shared_ptr<arma::mat> trainingY){
    int T = trainingY->n_elem;
    

    
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

// test the gradients by numerical gradients checking

void testGrad(char* filename) {
    
    std::shared_ptr<arma::mat> trainingX(new arma::mat);
    std::shared_ptr<arma::mat> trainingY(new arma::mat);
    int T = 2;
    trainingX->zeros(1, T);
     trainingY->ones(1, T);
    /* RNN constructor parameters passed as:
        RNN(int numHiddenLayers0, int hiddenLayerInputDim0,
        int hiddenLayerOutputDim0, int inputDim0, int outputDim0, 
        std::shared_ptr<arma::mat> trainingX0, std::shared_ptr<arma::mat> trainingY0)
     */
    NeuralNetParameter message; 
    ReadProtoFromTextFile(filename, &message);
    RNN rnn(message);
    rnn.setTrainingSamples(trainingX, trainingY);
    // before applying the LSTM backprop model, generate numerical gradients by just forward pass.
    rnn.calNumericGrad();

    std::vector<std::shared_ptr<arma::mat>> gradVec;
    rnn.calGradient();
    gradVec = rnn.netGradients();
    
    for (int i = 0; i < gradVec.size(); i++){
        std::cout << i << std::endl;
        gradVec[i]->print();
    }
    
    
}


#if 0
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

#endif
