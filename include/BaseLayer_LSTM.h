#pragma once

#include <vector>
#include "BaseLayer.h"


namespace NeuralNet{
   
struct BaseLayer_LSTM: public BaseLayer {
   
        BaseLayer_LSTM(int inputDim0, int outputDim0, ActivationType actType0,
                std::shared_ptr<Initializer> init_W = nullptr, std::shared_ptr<Initializer> init_B = nullptr,
				bool dropout = false, double dropr = 0.3)
                : BaseLayer(inputDim0, outputDim0, actType0, init_W, init_B, dropout, dropr){
                  }
         
    int timeLength;
    // for LSTM layer, each has to record the input to the layer at each time, which is 
       // used for caculating the gradient in backpropagation
public:
    std::vector<std::shared_ptr<arma::mat>> inputMem;
    std::vector<std::shared_ptr<arma::mat>> outputMem;
    
    // save inputs at all time points during the LSTM forward pass
    void saveInputMemory();
    // save outputs at all time points during the LSTM forward pass
    void saveOutputMemory();
    
    // extract out the specific input or output at time point t during backpropagation
     // to calculate the gradient
    std::shared_ptr<arma::mat> getInputMemory(int t);
    std::shared_ptr<arma::mat> getOutputMemory(int t);
    
    // @ override
    virtual void accumulateGrad(std::shared_ptr<arma::mat> delta_in, int t);
    // @ override
    virtual void calGrad(std::shared_ptr<arma::mat> delta_in, int timePoint);
    
    void clearAccuGrad();
};

}
