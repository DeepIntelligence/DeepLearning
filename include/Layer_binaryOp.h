#pragma once
#include "common.h"

namespace NeuralNet{

struct Layer_binaryOp : public Layer{
public:
	virtual ~Layer_binaryOp(){}
	Layer_binaryOp(){}
	Layer_binaryOp(int inputOneDim0, int inputTwoDim0, int outputDim0):Layer(outputDim0), inputOneDim(inputOneDim0), inputTwoDim(inputTwoDim0){}
	// save inputs at all time points during the LSTM forward pass
    virtual void saveInputMemory();

	std::shared_ptr<arma::mat> inputOne, inputTwo;
	std::shared_ptr<arma::mat> delta_outOne, delta_outTwo;
	int inputOneDim, inputTwoDim;
	std::vector<std::shared_ptr<arma::mat>> inputOneMem, inputTwoMem;

};

inline void Layer_binaryOp::saveInputMemory(){
    inputOneMem.push_back(inputOne);
    inputTwoMem.push_back(inputTwo);
}

}
