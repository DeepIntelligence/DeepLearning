#pragma once
#include "common.h"

namespace NeuralNet{

struct Layer_unitaryOp : public Layer{
public:
	virtual ~Layer_unitaryOp(){}
	Layer_unitaryOp(){}
	Layer_unitaryOp(int inputDim0, int outputDim0):Layer(outputDim0), inputDim(inputDim0){}
	    // save inputs at all time points during the LSTM forward pass
    virtual void saveInputMemory();
	virtual void setInput(std::shared_ptr<arma::mat> input0){ input = input0;}
	virtual std::shared_ptr<arma::mat> getDelta_out() {return delta_out;} 
	std::shared_ptr<arma::mat> input;
	std::shared_ptr<arma::mat> delta_out;
	int inputDim;
	std::vector<std::shared_ptr<arma::mat>> inputMem;

};

inline void Layer_unitaryOp::saveInputMemory(){
    inputMem.push_back(input);
}
}
