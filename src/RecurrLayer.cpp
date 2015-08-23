#include "RecurrLayer.h"

using namespace NeuralNet;

RecurrLayer::RecurrLayer(int in1, int in2, int out, ActivationType actType0, std::shared_ptr<Initializer> init_W_one, std::shared_ptr<Initializer> init_W_two,
        std::shared_ptr<Initializer> init_B):
        MultiAddLayer(in1,in2,out,actType0,init_W_one,init_W_two,init_B){
          
	output_prev = std::make_shared<arma::mat>(); 
	delta_outTwo_prev = std::make_shared<arma::mat>();
}

void RecurrLayer::savePrevOutput(){
	*output_prev = *output;
}
void RecurrLayer::savePrevDelta_outTwo(){
	*delta_outTwo_prev = *delta_outTwo;
}
