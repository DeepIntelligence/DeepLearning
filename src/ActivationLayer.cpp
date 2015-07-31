#include "ActivationLayer.h"

namespace NeuralNet{


void ActivationLayer::activateUp(){
     (*output) = *input;
	output->transform([](double val){return 1.0/(1.0+exp(-val));}); 
};

void ActivationLayer::updatePara(std::shared_ptr<arma::mat> delta_in){
	delta_out = std::make_shared<arma::mat>();
	arma::mat deriv = (1 - (*output)) % (*output);        
	(*delta_out) = (*delta_in) % deriv;
}
}

