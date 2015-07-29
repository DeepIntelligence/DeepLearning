#pragma once
#include <memory>
#include <armadillo>
#include <iostream>
#include <vector>
#include "../LSTM/BaseLayer_LSTM.h"

namespace NeuralNet {

    class RNN {
        
    public:
        RNN(int numHiddenLayers0, int hiddenLayerInputDim0,
        int hiddenLayerOutputDim0, int inputDim0, int outputDim0, 
        std::shared_ptr<arma::mat> trainingX0, std::shared_ptr<arma::mat> trainingY0);
        void forward();
        void backward();
        void train();
        void savePara(std::string filename); // try to save all the parameters in the LSTM for further use
        void test();
        void calNumericGrad();
    private:
        std::vector<BaseLayer_LSTM> hiddenLayers;
        BaseLayer_LSTM* netOutputLayer;
        std::shared_ptr<arma::mat> trainingY, trainingX;
        int numHiddenLayers, hiddenLayerInputDim, hiddenLayerOutputDim;
        int rnnInputDim, rnnOutputDim;
        
        

    };

}


