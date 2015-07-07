#include "ElementwiseLayer.h"

namespace NeuralNet{


void ElementwiseLayer::activateUp(){
     // elementwise product
     (*output) = (*inputOne) % (*inputTwo); 

};

void ElementwiseLayer::updatePara(std::shared_ptr<arma::mat> delta_in){

	(*delta_outOne) = (*inputTwo) * (*delta_in);
	(*delta_outTwo) = (*inputOne) * (*delta_in);

}
}

