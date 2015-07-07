#pragma once
#include <memory>
#include <armadillo>
#include <vector>
#include "../BaseLayer/BaseLayer.h"
#include "../ElementwiseLayer/ElementwiseLayer.h"


namespace NeuralNet {

    class RNN_LSTM {
    public:
        RNN_LSTM(int numHiddenLayers0, int hiddenLayerInputDim0,
        int hiddenLayerOutputDim0, int inputDim0, int outputDim0, 
        std::shared_ptr<arma::mat> trainingX0, std::shared_ptr<arma::mat> trainingY0);
        void forward();
        void backward();
    private:
        std::vector<BaseLayer> inGateLayers, forgetGateLayers, cellStateLayers, outputGateLayers, informationLayers;
        std::vector<ElementwiseLayer> outputLayers, forgetElementGateLayers, inputElementGateLayers;
        
        std::vector<BaseLayer> layerOutput_prev, cellState_prev;
        std::shared_ptr<arma::mat> trainingY, trainingX, netOutput;
        int numHiddenLayers, hiddenLayerInputDim, hiddenLayerOutputDim;
        int rnnInputDim, rnnOutputDim;
        

    };

}


