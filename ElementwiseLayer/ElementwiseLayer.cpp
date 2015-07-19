#include "ElementwiseLayer.h"

namespace NeuralNet{


void ElementwiseLayer::activateUp(){
     // elementwise product
     (*output) = (*inputOne) % (*inputTwo); 

};
// save inputs at all time points during the LSTM forward pass
void ElementwiseLayer::saveInputMemory() {
    inputMemOne.push_back(inputOne);
    inputMemTwo.push_back(inputTwo);
}

// extract out the specific input at time point t during backpropagation
// to calculate the gradient
//std::shared_ptr<arma::mat> ElementwiseLayer::getInputMemory(int t) {
//    return inputMem[t];
//}
void ElementwiseLayer::updatePara(std::shared_ptr<arma::mat> delta_in, int timePoint){

	(*delta_outOne) = (*inputMemTwo[timePoint]) * (*delta_in);
	(*delta_outTwo) = (*inputMemOne[timePoint]) * (*delta_in);

}
}

