#pragma once
#include <memory>
#include <armadillo>
#include <vector>
#include "../BaseLayer/BaseLayer.h"
#include "../ElementwiseLayer/ElementwiseLayer.h"


namespace NeuralNet {

    class RNN_LSTM {
    public:
        forward();
        backward();
    private:
        std::vector<BaseLayer> inGateLayers, forgetGateLayers, cellStateLayers, outputGateLayers, informationLayers;
        std::vector<ElementwiseLayer> outputLayers, forgetElementGateLayers, inputElementGateLayers;
        
        std::vector<BaseLayer> layerOutput_prev, cellState_prev;
        
        int numHiddenLayers, hiddenLayerInputDim, hiddenLayerOutputDim;
        int rnnInputDim, rnnOutputDim;


    }

}

