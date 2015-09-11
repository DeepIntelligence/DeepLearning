#include "ElementwiseLayer.h"

namespace NeuralNet{


void ElementwiseLayer::activateUp(){
     // elementwise product
     (*output) = (*inputOne) % (*inputTwo); 
};

void ElementwiseLayer::calGrad(std::shared_ptr<arma::mat> delta_in){
	(*delta_outOne) = (*inputTwo) % (*delta_in);
	(*delta_outTwo) = (*inputOne) % (*delta_in);

}

void ElementwiseLayer::calGrad(std::shared_ptr<arma::mat> delta_in, int timePoint){
	(*delta_outOne) = (*inputTwoMem[timePoint]) % (*delta_in);
	(*delta_outTwo) = (*inputOneMem[timePoint]) % (*delta_in);

}
}

