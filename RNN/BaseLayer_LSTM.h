#pragma once

#include <vector>
#include "../BaseLayer/BaseLayer.h"


namespace NeuralNet{
   
struct BaseLayer_LSTM: public BaseLayer {
   
        BaseLayer_LSTM(int inputDim0, int outputDim0, ActivationType actType0,
                bool dropout = false, double dropr = 0.3, int T=0)
                : BaseLayer(inputDim0, outputDim0, actType0, dropout, dropr),
                timeLength(T){
                  }
        
    int timeLength;
    // for LSTM layer, each has to record the input to the layer at each time, which is 
       // used for caculating the gradient in backpropagation
public:
    std::vector<std::shared_ptr<arma::mat>> inputMem;
    
    // save inputs at all time points during the LSTM forward pass
    void saveInputMemory();
    
    // extract out the specific input at time point t during backpropagation
     // to calculate the gradient
    std::shared_ptr<arma::mat> getInputMemory(int t);
    
    // @ override
    virtual void accumulateGrad(std::shared_ptr<arma::mat> delta_in, int t);
    // @ override
    virtual void calGrad(std::shared_ptr<arma::mat> delta_in, int timePoint);
    
    void clearAccuGrad();
};

}