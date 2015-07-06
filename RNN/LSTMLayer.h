#pragma once
#include "../BaseLayer/BaseLayer.h"

namespace NeuralNet {

    class RNN_LSTM {
    public:
        forward();
        backward();
    private:
        std::vector<BaseLayer> inGateLayers, forgetGateLayers, cellStateLayers, outputGateLayers, informationLayers;
        std::vector<ElementWiseLayer> outputLayers, forgetElementGateLayers, inputElementGateLayers;

        int numHiddenLayers, hiddenLayerInputDim, hiddenLayerOutputDim;
        int inputDim, outputDim;


    }

}

