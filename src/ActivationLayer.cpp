#include "ActivationLayer.h"

namespace NeuralNet{


void ActivationLayer::activateUp(){
	output = input;
    ApplyActivation(input, actType); 
};

void ActivationLayer::calGrad(std::shared_ptr<arma::mat> delta_in){
	delta_out = delta_in;
	GetActivationGradient(delta_in, delta_out, actType);
}

}

