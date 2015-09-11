#include "ElmanRL.h"

using namespace NeuralNet;
using namespace DeepLearning;

ElmanRL::ElmanRL(NeuralNetParameter neuralNetPara0):RNN(neuralNetPara0){
}

arma::mat ElmanRL::forwardInTime(std::shared_ptr<arma::mat> input) {
    std::shared_ptr<arma::mat> commonInput(new arma::mat);
    if (this->time == 0) {
        for (int l = 0; l < numRecurrLayers; l++) {
            (recurrLayers[l].getPrevOutput())->zeros(recurrLayerOutputDim, 1);
            recurrLayers[l].inputOneMem.clear();
            recurrLayers[l].inputTwoMem.clear();
            recurrLayers[l].outputMem.clear();
        }
        for (int l = 0; l < numBaseLayers; l++) {
        	baseLayers[l].inputMem.clear();
        	baseLayers[l].outputMem.clear();
    	}
    }
    for (int l = 0; l < numRecurrLayers; l++) {
    	recurrLayers[l].inputOne = std::shared_ptr<arma::mat>(new arma::mat(*(recurrLayers[l].getPrevOutput())));
  
        if (l == 0) {
        	
			recurrLayers[l].inputTwo = std::shared_ptr<arma::mat>(new arma::mat(input->rows(0, rnnInputDim - 1)));
        } else {
			recurrLayers[l].inputTwo = std::shared_ptr<arma::mat>(new arma::mat(*(recurrLayers[l - 1].output)));
        }
        recurrLayers[l].activateUp();
#if 0
        recurrLayers[l].W_one->print("W_one");
        recurrLayers[l].W_two->print("W_two");
        recurrLayers[l].B->print("B");
        recurrLayers[l].inputOne->print("input one");
        recurrLayers[l].inputTwo->print("input two");
        recurrLayers[l].output->print("output");
#endif        
    }
    	
    for (int l = 0; l < numBaseLayers; l++) {
       	if (l == 0) {
       		arma::mat action(input->rows(rnnInputDim, input->n_rows - 1));
                *commonInput = arma::join_cols(*(recurrLayers[numRecurrLayers-1].output), action);
    		baseLayers[l].input = commonInput;   	
       	} else {
       		baseLayers[l].input = baseLayers[l - 1].output;
       	}
       	baseLayers[l].activateUp();
    }
    return *(baseLayers[numBaseLayers - 1].output);
}
 
void ElmanRL::backward() {
   
    std::shared_ptr<arma::mat> delta(new arma::mat);
    for (int l = 0; l < numRecurrLayers; l++) {
        recurrLayers[l].clearAccuGrad();
    }
    for (int l = 0; l < numBaseLayers; l++) {
        baseLayers[l].clearAccuGrad();
    }      
    int T = trainingY->n_cols;
    for (int t = T - 1; t >= 0; t--){
    // the top most layer from target - network's output
        *delta = *(baseLayers[numBaseLayers - 1].outputMem[t]) - trainingY->col(t);    	
    	for (int l = numBaseLayers - 1; l >=0; l--) {
    		baseLayers[l].accumulateGrad(delta, t);
    		*delta = *(baseLayers[l].delta_out);
    	}
        for (int l = numRecurrLayers - 1; l >= 0; l--){
            // delta error from the same time, propagate from upper layer to lower layer 
            if (l == numRecurrLayers - 1){ 
                *delta = baseLayers[0].delta_out->rows(0, recurrLayerOutputDim - 1); 
            }else{ 
                *delta = *(recurrLayers[l+1].delta_outTwo);                
            }   
            
            if (t < T - 1) {
               *delta += *(recurrLayers[l].getPrevDeltaOutOne());
            }              
            // so far, the generated delta error is for the output h of each layer at each time
            recurrLayers[l].accumulateGrad(delta, t);
            recurrLayers[l].savePrevDeltaOutOne();
        }
	}  
}

