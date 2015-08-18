#include "LinearAdditionLayer.h"

namespace NeuralNet{


void LinearAdditionLayer::activateUp(){
     (*output) = (*inputOne) + (*inputTwo); 
};

void LinearAdditionLayer::calGrad(std::shared_ptr<arma::mat> delta_in){
	delta_outOne = delta_in;
	delta_outTwo = delta_in;
}

}

