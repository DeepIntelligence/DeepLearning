//#include "RNN.h"
#include <armadillo>
#include <vector>
#include "../BaseLayer/BaseLayer.h"

RNN_LSTM::RNN_LSTM(int numHiddenLayers0, int hiddenLayerInputDim0,
        int hiddenLayerOutputDim0, int inputDim0, int outputDim0) {


    numHiddenLayers = numHiddenLayers0;
    hiddenLayerInputDim = hiddenLayerInputDim0;
    hiddenLayerOutputDim0 = hiddenLayerOutputDim0;
    rnnInputDim = inputDim0;
    rnnOutputDim = outputDim0;


    
    
// now initialize the two vectors
    for (int i = 0; i < numHiddenLayers; i++){
        
        // i=0 is the first hidden layer with input of data input and last time hidden
        if (i == 0) {
            
            int inputDim = rnnInputDim + hiddenLayerOutputDim0;
            int outputDim = hiddenLayerOutputDim0;
            
            inGateLayers.push_back(BaseLayer(inputDim, outputDim, BaseLayer::sigmoid));    
            forgetGateLayers.push_back(BaseLayer(inputDim, outputDim, BaseLayer::sigmoid));
            cellStateLayers.push_back(BaseLayer(inputDim, outputDim, BaseLayer::sigmoid));
            outputGateLayers.push_back(BaseLayer(inputDim, outputDim, BaseLayer::sigmoid));
            informationLayers
        }
        inGateLayers.push_back(BaseLayer(inputDim, outputDim, BaseLayer::sigmoid));    
        forgetGateLayers.push_back(BaseLayer(inputDim))
        cellStateLayers.
        outputGateLayers
        informationLayers
    }
    


}

RNN_LSTM::forward() {

    layerOutput.output->zeros();
    // to forward pass the Deep LSTM model, loop each time point, at each time, go through bottom layer to top layer
    for (int t = 0; t < T; t++){
        for (int l = 0; l < numHiddenLayers; l++){

            if (l == 0) {
                lowerLayer = dataLayer;
            } else {
                lowerLayer = layers[l - 1];
            }

    // concatenate to a large vector
    commonInput = [ lowerLayer.output;
    layerOutput_prev[l].output];
    //1
    inGateLayers.input = commonInput
            inGateLayers.activatUp();
    //2
    InformationLayers[l].input = commonInput
            InformationLayers[l].activateUp();
    //3
    inputElementGateLayers[l].inputOne = InformationLayers.output;
    inputElementGateLayers[l].inputTwo = inGateLayers.output;
    inputElementGateLayers[l].activateUp();

    //4
    forgetGateLayers[l].input = commonInput;
    forgetGateLayers[l].activateUp();
    //5
    forgetElementGate[l].inputOne = forgetGate.output;
    forgetElementGate[l].inputTwo = cellStateLayer.output;
    forgetElementGate[l].activateUp();
    //6
    cellState[l].input = inputElementGate.output + forgetElementGate.output;
    cellState_prev[l].input = cellState[l].input;
    cellState[l].activateUp();
    //7
    outputGate[l].input = commonInput;
    outputGate[l].activateUp();
    //8
    outputLayer[l].inputOne = outputGate.output;
    outputLayer[l].inputTwo = cellState.output;
    outputLayer[l].activateUp();

    layerOutput_prev[l] = layerOutput[l];
}
}

}

RNN_LSTM::backward() {
    layerOutput.output->zeros();

    // to backprop or backpass the Deep LSTM, start from the top layer of the last time point T, 
    // and then go through from top to bottom, and then go to previous time point, and loop 
    // from top to bottom layers again
    for (int t = T - 1; t >= T; t++){
        for (int l = L - 1; l >= 0; l++){

            if (l == 0) {
                lowerLayer = dataLayer;
            } else {
                lowerLayer = layers[l - 1];
            }

    delta = y - y_target;
    // this layer's output error comes from last time's (1)inGate, (2)g, (4)forgetGate, (7)outputGate, since output of each hidden layer will
    // be the input for the inGate, g, forgetGate, outputGate
    delta_prev = inGate_next.delta + Information_next.delta + forgetGate_next.delta + outputGate_next.delta;

    delta += delta_prev

            outputLayer[l].updatePara(delta);
    outputGate[l].updatePara(outputLayer.deltaoutOne);
    //	cellSate[l].deltaOut +=cellState_next[l].deltaOut;
    //	cellSate[l].deltaOut +=forgetElementGate_prev[l].deltaOut;
    cellState[l].updatePara(outputLayer.deltaoutTwo);
    inputElementGate.updatePara(cellState.deltaOut);

}
}

}


