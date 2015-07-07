#include "LSTMLayer.h"

using namespace NeuralNet;

RNN_LSTM::RNN_LSTM(int numHiddenLayers0, int hiddenLayerInputDim0,
        int hiddenLayerOutputDim0, int inputDim0, int outputDim0, 
        std::shared_ptr<arma::mat> trainingX0, std::shared_ptr<arma::mat> trainingY0) {


    // at beginning, we assume all the hidden layers have the same size, 
    numHiddenLayers = numHiddenLayers0;
    hiddenLayerInputDim = hiddenLayerInputDim0;
    hiddenLayerOutputDim0 = hiddenLayerOutputDim0;
    rnnInputDim = inputDim0;
    rnnOutputDim = outputDim0;


    
    
// now initialize the two vectors
    for (int i = 0; i < numHiddenLayers; i++){
        
        // i=0 is the first hidden layer with input consisting 
          // of data input and last time hidden
        if (i == 0) {
            
            int inputDim = rnnInputDim + hiddenLayerOutputDim0;
            int outputDim = hiddenLayerOutputDim0;
            
            inGateLayers.push_back(BaseLayer(inputDim, outputDim, BaseLayer::sigmoid));    
            forgetGateLayers.push_back(BaseLayer(inputDim, outputDim, BaseLayer::sigmoid));
            cellStateLayers.push_back(BaseLayer(inputDim, outputDim, BaseLayer::sigmoid));
            outputGateLayers.push_back(BaseLayer(inputDim, outputDim, BaseLayer::sigmoid));
            informationLayers.push_back(BaseLayer(inputDim, outputDim, BaseLayer::sigmoid));
            
            inputElementGateLayers.push_back(ElementwiseLayer());
            forgetElementGateLayers.push_back(ElementwiseLayer());
            outputLayers.push_back(ElementwiseLayer());
        }
        else if(i== numHiddenLayers-1){
            int inputDim = hiddenLayerOutputDim0 + hiddenLayerOutputDim0;
            int outputDim = rnnOutputDim;
            
            inGateLayers.push_back(BaseLayer(inputDim, outputDim, BaseLayer::sigmoid));    
            forgetGateLayers.push_back(BaseLayer(inputDim, outputDim, BaseLayer::sigmoid));
            cellStateLayers.push_back(BaseLayer(inputDim, outputDim, BaseLayer::sigmoid));
            outputGateLayers.push_back(BaseLayer(inputDim, outputDim, BaseLayer::sigmoid));
            informationLayers.push_back(BaseLayer(inputDim, outputDim, BaseLayer::sigmoid));
            
            inputElementGateLayers.push_back(ElementwiseLayer());
            forgetElementGateLayers.push_back(ElementwiseLayer());
            outputLayers.push_back(ElementwiseLayer());
        }
        // i!=0 all other layers have the input consisting 
          // of hidden output from lower layer at the same time and
           // hidden output from same layer but at previous time
        else{
            
            //
            int inputDim = hiddenLayerOutputDim0 + hiddenLayerOutputDim0;
            int outputDim = hiddenLayerOutputDim0;
            
            inGateLayers.push_back(BaseLayer(inputDim, outputDim, BaseLayer::sigmoid));    
            forgetGateLayers.push_back(BaseLayer(inputDim, outputDim, BaseLayer::sigmoid));
            cellStateLayers.push_back(BaseLayer(inputDim, outputDim, BaseLayer::sigmoid));
            outputGateLayers.push_back(BaseLayer(inputDim, outputDim, BaseLayer::sigmoid));
            informationLayers.push_back(BaseLayer(inputDim, outputDim, BaseLayer::sigmoid));
            
            inputElementGateLayers.push_back(ElementwiseLayer());
            forgetElementGateLayers.push_back(ElementwiseLayer());
            outputLayers.push_back(ElementwiseLayer());
            
        }
        
    }
    
    netOutput = std::make_shared<arma::mat>();

}

void RNN_LSTM::forward() {

    std::shared_ptr<arma::mat> commonInput(new arma::mat);
    arma::mat outputLayers_prev_output[numHiddenLayers];
    arma::mat cellStateLayers_prev[numHiddenLayers];
    // Deep LSTM
    //layerOutput.output->zeros();
    // to forward pass the Deep LSTM model, loop each time point, 
     // at each time, go through bottom layer to top layer
    int T = trainingY->n_cols;
    for (int t = 0; t < T; t++){
        for (int l = 0; l < numHiddenLayers; l++) {
    // concatenate to a large vector            
            if (l == 0) {
                *commonInput = arma::join_cols(outputLayers_prev_output[l], trainingX->col(t));
            } else {
                *commonInput = arma::join_cols(outputLayers_prev_output[l], *(outputLayers[l-1].output));
            }

    //1
        inGateLayers[l].input = commonInput;
        inGateLayers[l].activateUp();
    //2
        informationLayers[l].input = commonInput;
        informationLayers[l].activateUp();
    //3
        inputElementGateLayers[l].inputOne = informationLayers[l].output;
        inputElementGateLayers[l].inputTwo = inGateLayers[l].output;
        inputElementGateLayers[l].activateUp();

    //4
        forgetGateLayers[l].input = commonInput;
        forgetGateLayers[l].activateUp();
    //5
        forgetElementGateLayers[l].inputOne = forgetGateLayers[l].output;
        forgetElementGateLayers[l].inputTwo = cellStateLayers[l].output;
        forgetElementGateLayers[l].activateUp();
    //6
        (*cellStateLayers[l].input) = *(inputElementGateLayers[l].output) + *(forgetElementGateLayers[l].output);
        cellStateLayers_prev[l]= *(cellStateLayers[l].input);
        cellStateLayers[l].activateUp();
    //7
        outputGateLayers[l].input = commonInput;
        outputGateLayers[l].activateUp();
    //8
        outputLayers[l].inputOne = outputGateLayers[l].output;
        outputLayers[l].inputTwo = cellStateLayers[l].output;
        outputLayers[l].activateUp();

    if(l == numHiddenLayers-1){
//        if (mask(t)) {
//            netOutput->col(t).zeros();
//        } else {
            netOutput->col(t) = *(outputLayers[l].output);
//        }
    }
    
    outputLayers_prev_output[l] = *(outputLayers[l].output);
}
}

}
#if 0
RNN_LSTM::backward() {
//    layerOutput.output->zeros();

    // to backprop or backpass the Deep LSTM, start from the top layer of the last time point T, 
    // and then go through from top to bottom, and then go to previous time point, and loop 
    // from top to bottom layers again
    for (int t = T - 1; t >= T; t++){
        for (int l = numHiddenLayers - 1; l >= 0; l++){

            if (l == numHiddenLayers - 1){
                delta = netOutput->col(t) - trainingY->col(t);
            } 
    
    // this layer's output error comes from last time's (1)inGate, (2)g, (4)forgetGate, (7)outputGate, since output of each hidden layer will
    // be the input for the inGate, g, forgetGate, outputGate
            if (t < T-1) {
                delta_prev = inGate_next_deltaOut[l] + Information_next_deltaOut[l] + 
                    forgetGate_next_deltaOut[l] + outputGate_next_deltaOut[l];
                delta += delta_prev;
            }    
   //8         
            outputLayers[l].updatePara(delta);
   //7
            outputGateLayers[l].updatePara(outputLayer.deltaoutOne);
    //	cellSate[l].deltaOut +=cellState_next[l].deltaOut;
    //	cellSate[l].deltaOut +=forgetElementGate_prev[l].deltaOut;
   //6  cellStateLayers.delta_in = (5) cellState_next_deltaIn + (8) outputLayer.deltaoutTwo 
            cellStateLayers[l].updatePara(cellState_next_deltaIn[l]+forgetElementGateLayers[l].delta_outTwo);
//    inputElementGate.updatePara(cellState.deltaOut);
    //5  forgetGateLayers[l].delta_in = cellStateLayers.delta_in;
            forgetGateLayers[l].updatePara(forgetElementGateLayers[l].delta_outOne);
            
    //3  inputElementGateLayers[l].delta_in = cellStateLayers.delta_in;
            inputElementGateLayers[l].updatePara(cellStateLayers.delta_in);
    //2
            inputGateLayers[l].updatePara(inputElementGateLayers[l].delta_outOne);
    //1        
            inputGateLayers[l].updatePara(inputElementGateLayers[l].delta_outTwo);

    //1        
    inGate_next_deltaOut[l] = *(inGateLayers[l].deltaOut);
    //4
    forgetGate_next_deltaOut[l] = *(forgetGateLayers[l].deltaOut);
    //2
    information_next_deltaOut[l] = *(informationLayers[l].deltaOut);
    //7
    outputGate_next_deltaOut[l] = *(outputGate[l].deltaOut);
    //5
    cellState_next_deltaOut[l] = *(cellStateLayers[l].deltaIn);
        }
    

}

}

#endif




