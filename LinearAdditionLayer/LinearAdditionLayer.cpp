#include "LinearAdditionLayer.h"

namespace NeuralNet{


void LinearAdditionLayer::activateUp(){
     (*output) = (*inputOne) + (*inputTwo); 
};

void LinearAdditionLayer::updatePara(std::shared_ptr<arma::mat> delta_in){

	delta_out = std::make_shared<arma::mat>(*delta_in);
}
}

