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


    
    
    /*
     * We keep each cell has 8 layers, including ...
     * now initialize the LSTM model based on unrolled inputs and outputs
     * deep LSTM with multiple layers initialization,
     * 
     */
     
    for (int i = 0; i < numHiddenLayers; i++){
        
        // i=0 is the first hidden layer with input consisting 
          // of data input and last time hidden
        if (i == 0) {
            
            // inputDim is the unrolled LSTM input for various layers, the same to outputDim
            int inputDim = rnnInputDim + hiddenLayerOutputDim0;
            int outputDim = hiddenLayerOutputDim0;
            
            inGateLayers.push_back(BaseLayer_LSTM(inputDim, outputDim, BaseLayer::sigmoid));    
            forgetGateLayers.push_back(BaseLayer_LSTM(inputDim, outputDim, BaseLayer::sigmoid));
            
            outputGateLayers.push_back(BaseLayer_LSTM(inputDim, outputDim, BaseLayer::sigmoid));
            informationLayers.push_back(BaseLayer_LSTM(inputDim, outputDim, BaseLayer::sigmoid));
            
            inputElementGateLayers.push_back(ElementwiseLayer());
            forgetElementGateLayers.push_back(ElementwiseLayer());
            outputElementLayers.push_back(ElementwiseLayer());
            
            cellLinearAdditionLayers.push_back(LinearAdditionLayer());
            cellStateActivationLayers.push_back(ActivationLayer());
        }
        // i!=0 all other layers have the input consisting 
          // of hidden output from lower layer at the same time and
           // hidden output from same layer but at previous time
        else if(i < numHiddenLayers-1){
              
            // inputDim is the unrolled LSTM input for various layers, the same to outputDim
            int inputDim = hiddenLayerOutputDim0 + hiddenLayerOutputDim0;
            int outputDim = hiddenLayerOutputDim0;
            
            inGateLayers.push_back(BaseLayer_LSTM(inputDim, outputDim, BaseLayer::sigmoid));    
            forgetGateLayers.push_back(BaseLayer_LSTM(inputDim, outputDim, BaseLayer::sigmoid));
    
            outputGateLayers.push_back(BaseLayer_LSTM(inputDim, outputDim, BaseLayer::sigmoid));
            informationLayers.push_back(BaseLayer_LSTM(inputDim, outputDim, BaseLayer::sigmoid));
            
            inputElementGateLayers.push_back(ElementwiseLayer());
            forgetElementGateLayers.push_back(ElementwiseLayer());
            outputElementLayers.push_back(ElementwiseLayer());
            cellLinearAdditionLayers.push_back(LinearAdditionLayer());
            cellStateActivationLayers.push_back(ActivationLayer());
        
        }else if(i== numHiddenLayers-1){
            // inputDim is the unrolled LSTM input for various layers, the same to outputDim
            int inputDim = hiddenLayerOutputDim0 + hiddenLayerOutputDim0;
            int outputDim = rnnOutputDim;
            
            inGateLayers.push_back(BaseLayer_LSTM(inputDim, outputDim, BaseLayer::sigmoid));    
            forgetGateLayers.push_back(BaseLayer_LSTM(inputDim, outputDim, BaseLayer::sigmoid));
            
            outputGateLayers.push_back(BaseLayer_LSTM(inputDim, outputDim, BaseLayer::sigmoid));
            informationLayers.push_back(BaseLayer_LSTM(inputDim, outputDim, BaseLayer::sigmoid));
            
            inputElementGateLayers.push_back(ElementwiseLayer());
            forgetElementGateLayers.push_back(ElementwiseLayer());
            outputElementLayers.push_back(ElementwiseLayer());
            
            cellLinearAdditionLayers.push_back(LinearAdditionLayer());
            cellStateActivationLayers.push_back(ActivationLayer());
        }
        
        
        
        
        }
            netOutput = std::make_shared<arma::mat>();
 }
    




void RNN_LSTM::forward() {

    std::shared_ptr<arma::mat> commonInput(new arma::mat);
    arma::mat outputLayers_prev_output[numHiddenLayers];
    arma::mat cellStateLayers_prev_output[numHiddenLayers];
    arma::mat cellStateOutput;
    for (int l = 0; l < numHiddenLayers; l++){
        cellStateLayers_prev_output[l].zeros();
        outputLayers_prev_output[l].zeros();
        
    }
    
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
                *commonInput = arma::join_cols(outputLayers_prev_output[l], *(outputElementLayers[l-1].output));
            }

    //1
        inGateLayers[l].input = commonInput;
        inGateLayers[l].saveInputMemory();
        inGateLayers[l].activateUp();
    //2
        informationLayers[l].input = commonInput;
        informationLayers[l].saveInputMemory();
        informationLayers[l].activateUp();
    //3
        inputElementGateLayers[l].inputOne = informationLayers[l].output;
        inputElementGateLayers[l].inputTwo = inGateLayers[l].output;
        inputElementGateLayers[l].saveInputMemory();
        inputElementGateLayers[l].activateUp();

    //4
        forgetGateLayers[l].input = commonInput;
        forgetGateLayers[l].saveInputMemory();
        forgetGateLayers[l].activateUp();
    //5
        forgetElementGateLayers[l].inputOne = forgetGateLayers[l].output;
    // TODO here should avoid duplication of memory    
        forgetElementGateLayers[l].inputTwo = std::make_shared<arma::mat>(cellStateLayers_prev_output[l]);
        forgetElementGateLayers[l].saveInputMemory();
        forgetElementGateLayers[l].activateUp();
    //6
        cellLinearAdditionLayers[l].inputOne = inputElementGateLayers[l].output;
        cellLinearAdditionLayers[l].inputTwo = forgetElementGateLayers[l].output;
        cellLinearAdditionLayers[l].activateUp();
        
        cellStateLayers_prev_output[l]= *(cellLinearAdditionLayers[l].output);
    //6.5
        cellStateActivationLayers[l].input = cellLinearAdditionLayers[l].output;
        cellStateActivationLayers[l].activateUp();
    //7
        outputGateLayers[l].input = commonInput;
        outputGateLayers[l].saveInputMemory();
        outputGateLayers[l].activateUp();
    //8
        outputElementLayers[l].inputOne = outputGateLayers[l].output;
        outputElementLayers[l].inputTwo = cellStateActivationLayers[l].output;
        outputElementLayers[l].saveInputMemory();
        outputElementLayers[l].activateUp();

    if(l == numHiddenLayers-1){
//        if (mask(t)) {
//            netOutput->col(t).zeros();
//        } else {
            netOutput->col(t) = *(outputElementLayers[l].output);
//        }
    }
    
    outputLayers_prev_output[l] = *(outputElementLayers[l].output);
}
}

}
 
void RNN_LSTM::backward() {
//    layerOutput.output->zeros();

    // to backprop or backpass the Deep LSTM, start from the top layer of the last time point T, 
    // and then go through from top to bottom, and then go to previous time point, and loop 
    // from top to bottom layers 
   
    arma::mat inGate_upstream_deltaOut[numHiddenLayers]; 
    arma::mat information_upstream_deltaOut[numHiddenLayers];
    arma::mat forgetGate_upstream_deltaOut[numHiddenLayers]; 
    arma::mat outputGate_upstream_deltaOut[numHiddenLayers];
    arma::mat cellState_upstream_deltaOut[numHiddenLayers];
    std::shared_ptr<arma::mat> cellStateLayers_deltaIn(new arma::mat);
    
    std::shared_ptr<arma::mat> delta, delta_upstream; // temporal delta is a vector
    delta = std::make_shared<arma::mat>();
    
    double learningRate = 0.1;
    
    int T = trainingY->n_cols;
    for (int t = T - 1; t >= 0; t++){
        for (int l = numHiddenLayers - 1; l >= 0; l++){
          
            // delta error from the same time, propagate from upper layer to lower layer 
            if (l == numHiddenLayers - 1){ // the top most layer from target - network's output
                    *delta = netOutput->col(t) - trainingY->col(t);
            }else{ // lower layer's delta error come from 1)inGate, (2)g, (4)forgetGate, 
            // (7)outputGate of upper hidden layer
                *delta = *(inGateLayers[l+1].delta_out) + *(informationLayers[l+1].delta_out) +
                        *(forgetGateLayers[l+1].delta_out) + *(outputGateLayers[l+1].delta_out);
            }   
            // another layer's output error comes from last time's (1)inGate, (2)g, (4)forgetGate, 
            //(7)outputGate, since output of each hidden layer will
            // be the input for the inGate, g, forgetGate, outputGate
            // temporal storage of this delta from upstream but the same layer
            if (t < T - 1) {               
                *delta += inGate_upstream_deltaOut[l] + information_upstream_deltaOut[l] +
                        forgetGate_upstream_deltaOut[l] + outputGate_upstream_deltaOut[l]; 
            }  
            
            // so far, the generated delta error is for the output h of each layer at each time
            
   //8      backpropagate from output layer 
            outputElementLayers[l].updatePara(delta, t); // pass delta directly to the outputlayer h
   
   //7      outputGate
            outputGateLayers[l].accumulateGrad(outputElementLayers[l].delta_outOne, t);
    //	cellSate[l].deltaOut +=cellState_next[l].deltaOut;
    //	cellSate[l].deltaOut +=forgetElementGate_prev[l].deltaOut;
    //6.5
         cellStateActivationLayers[l].updatePara(outputElementLayers[l].delta_outTwo);   
    //6  cellStateLayers.delta_in = (5) cellState_next_deltaIn + (8) outputLayer.deltaoutTwo
            (*cellStateLayers_deltaIn) = cellState_upstream_deltaOut[l]+
         *(cellStateActivationLayers[l].delta_out);
            cellLinearAdditionLayers[l].updatePara(cellStateLayers_deltaIn);
//    inputElementGate.updatePara(cellState.deltaOut);
    //5
            forgetElementGateLayers[l].updatePara(cellStateLayers[l].delta_out, t);
    //4  forgetGateLayers[l].delta_in = cellStateLayers.delta_in;
            forgetGateLayers[l].accumulateGrad(forgetElementGateLayers[l].delta_outOne, t);
            
    //3  inputElementGateLayers[l].delta_in = cellStateLayers.delta_in;
            inputElementGateLayers[l].updatePara(cellStateLayers[l].delta_out, t);
    //2
            informationLayers[l].accumulateGrad(inputElementGateLayers[l].delta_outOne, t);
    //1        
            inGateLayers[l].accumulateGrad(inputElementGateLayers[l].delta_outTwo, t);

    //1        
            inGate_upstream_deltaOut[l] = *(inGateLayers[l].delta_out);
    //4
            forgetGate_upstream_deltaOut[l] = *(forgetGateLayers[l].delta_out);
    //2
            information_upstream_deltaOut[l] = *(informationLayers[l].delta_out);
    //7
            outputGate_upstream_deltaOut[l] = *(outputGateLayers[l].delta_out);
    //5
            cellState_upstream_deltaOut[l] = *(cellStateLayers[l].delta_out);
     
        }
    

}
    
    for (int l = numHiddenLayers - 1; l >= 0; l++){
        outputGateLayers[l].updatePara_accu(learningRate);
        forgetGateLayers[l].updatePara_accu(learningRate);
        informationLayers[l].updatePara_accu(learningRate);
        inGateLayers[l].updatePara_accu(learningRate);
    
    }
    
    
    
}



