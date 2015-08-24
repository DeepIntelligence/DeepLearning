#include "RNN.h"

using namespace NeuralNet;
using namespace DeepLearning;

RNN::RNN(NeuralNetParameter neuralNetPara0){

    neuralNetPara = neuralNetPara0;
    // at beginning, we assume all the recurr layers have the same size,
    numRecurrLayers = neuralNetPara.rnnstruct().numrecurrlayers();
    recurrLayerInputDim = neuralNetPara.rnnstruct().recurrlayeroutputdim();
    recurrLayerOutputDim = neuralNetPara.rnnstruct().recurrlayeroutputdim();
    numBaseLayers = neuralNetPara.layerstruct_size();
    
    rnnInputDim = neuralNetPara.rnnstruct().inputdim();
    rnnOutputDim = neuralNetPara.layerstruct(numBaseLayers - 1).outputdim();
    
    for (int i = 0; i < numRecurrLayers; i++){
        int inputOneDim, inputTwoDim;
		int outputDim;
		NeuralNetInitializerParameter  w_one_init, w_two_init, b_init;
        w_one_init = neuralNetPara.rnnstruct().init_w_one();
        w_two_init = neuralNetPara.rnnstruct().init_w_two();
        b_init = neuralNetPara.rnnstruct().init_b();       
        // i=0 is the first recurr layer with input consisting 
          // of data input and last time recurr
        if (i == 0) {           
            // inputDim is the unrolled LSTM input for various layers, the same to outputDim
            inputOneDim = recurrLayerOutputDim;
            inputTwoDim = rnnInputDim;
            outputDim = recurrLayerOutputDim;           
          // of recurr output from lower layer at the same time and
           // recurr output from same layer but at previous time
       } else if(i <= numRecurrLayers-1){
              
            // inputDim is the unrolled LSTM input for various layers, the same to outputDim
            inputOneDim = recurrLayerOutputDim;
			inputTwoDim = recurrLayerOutputDim;
            outputDim = recurrLayerOutputDim;            
       }
        
		recurrLayers.push_back(RecurrLayer(inputOneDim, inputTwoDim, outputDim, GetActivationType(neuralNetPara.rnnstruct().activationtype()),
		InitializerBuilder::GetInitializer(w_one_init), InitializerBuilder::GetInitializer(w_two_init), 
		InitializerBuilder::GetInitializer(b_init)));     
	}
	
	for (int i = 0; i < numBaseLayers; i++){
        NeuralNetInitializerParameter  w_init, b_init;
        w_init = neuralNetPara.layerstruct(i).init_w();
        b_init = neuralNetPara.layerstruct(i).init_b();
 
		baseLayers.push_back(BaseLayer(neuralNetPara.layerstruct(i).inputdim(),
        neuralNetPara.layerstruct(i).outputdim(), GetActivationType(neuralNetPara.layerstruct(i).activationtype()),
        InitializerBuilder::GetInitializer(w_init), InitializerBuilder::GetInitializer(b_init)));
	}
    
	fillNetGradVector();
}

void RNN::resetWeight(){
	for (int l = 0; l < numRecurrLayers; l++) {
        recurrLayers[l].initializeWeight();
    }    
	for (int l = 0; l < numBaseLayers; l++) {
        baseLayers[l].initializeWeight();
    }
}
void RNN::setTime(int t){ this->time = t;}
int RNN::getTime(){return this->time;}
void RNN::zeroTime(){this->time = 0;}

arma::mat RNN::forwardInTime(std::shared_ptr<arma::mat> input) {
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
			recurrLayers[l].inputTwo = std::shared_ptr<arma::mat>(new arma::mat(*input));
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
    		baseLayers[l].input = recurrLayers[numRecurrLayers-1].output;   	
       	} else {
       		baseLayers[l].input = baseLayers[l - 1].output;
       	}
       	baseLayers[l].activateUp();
    }
    return *(baseLayers[numBaseLayers - 1].output);
}
void RNN::saveLayerInputOutput(){
    for (int l = 0; l < numRecurrLayers; l++) {
        recurrLayers[l].saveInputMemory();
        recurrLayers[l].saveOutputMemory();
    }
    
    for (int l = 0; l < numBaseLayers; l++) {
        baseLayers[l].saveInputMemory();
        baseLayers[l].saveOutputMemory();
    }        
}

void RNN::updateInternalState(){
    for (int l = 0; l < numRecurrLayers; l++) {
    	recurrLayers[l].savePrevOutput();
    }
}

void RNN::forward() {
    int T = trainingX->n_cols; 
    std::shared_ptr<arma::mat> input(new arma::mat);
    for (this->time = 0; this->time < T; (this->time)++){
        *input = trainingX->col(this->time);
        this->forwardInTime(input);
        this->saveLayerInputOutput();
        this->updateInternalState();
    }
}
 
void RNN::backward() {
   
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
                *delta = *(baseLayers[0].delta_out); 
            }else{ 
                *delta = *(recurrLayers[l+1].delta_outTwo);                
            }   
            
            if (t < T - 1) {
               *delta += *(recurrLayers[l].getPrevDeltaOutOne());
            }              
            // so far, the generated delta error is for the output h of each layer at each time
            recurrLayers[l].accumulateGrad(delta, t);
            recurrLayers[l].savePrevDeltaOutOne();
//            recurrLayer_upstream_deltaOut[l] = *(recurrLayers[l].delta_outOne);
        }
	}  
}

void RNN::fillNetGradVector(){
    for (int i = 0; i < this->numRecurrLayers; i++){
        netGradVector.push_back(this->recurrLayers[i].grad_W_one_accu);
        netGradVector.push_back(this->recurrLayers[i].grad_W_two_accu);
        netGradVector.push_back(this->recurrLayers[i].grad_B_accu);
    }
    
    for (int i = 0; i < this->numBaseLayers; i++){
    netGradVector.push_back(this->baseLayers[i].grad_W_accu);
    netGradVector.push_back(this->baseLayers[i].grad_B_accu);
    } 

}

void RNN::applyUpdates(std::vector<std::shared_ptr<arma::mat>> inGradVector){
    for (int i = 0; i < this->numRecurrLayers; i++){
        *(recurrLayers[i].W_one) -= *(inGradVector[3*i]);
        *(recurrLayers[i].W_two) -= *(inGradVector[3*i+1]);
        *(recurrLayers[i].B) -= *(inGradVector[3*i+2]);
    }
    for (int i = 0; i < this->numBaseLayers; i++){
        *(baseLayers[i].W) -= *(inGradVector[numRecurrLayers*3 + 2*i]);
        *(baseLayers[i].B) -= *(inGradVector[numRecurrLayers*3 + 2*i + 1]);
    }    
}

void RNN::calGradient(){
        forward();
        backward();
}

double RNN::getLoss(){
//  for calcuating the total error
    double error = 0.0;
    arma::mat delta;

    for (int k = 0; k < trainingY->n_cols; k++) {
        delta = *((baseLayers[numBaseLayers - 1].outputMem)[k]) - trainingY->col(k);
        error += arma::as_scalar(delta.st() * delta); 
    }
    return error;
}

std::shared_ptr<arma::mat> RNN::netOutput(){
    netOutput_ = std::make_shared<arma::mat>(this->rnnOutputDim, trainingX->n_cols);
    for (int k = 0; k < trainingX->n_cols; k++) {
        double* ptr = netOutput_->colptr(k);
        for (int i = 0; i < rnnOutputDim; i++){
            *(ptr+i) = (baseLayers[numBaseLayers - 1].outputMem[k])->at(i);
        }
    }
    return netOutput_;
}


void RNN::save(std::string filename){
    for (int l=0;l<numRecurrLayers;l++){
    	std::stringstream tag;   
        tag << l;
        recurrLayers[l].save(filename+"_recurrLayer_"+tag.str());
    }
    for (int l=0;l<numBaseLayers;l++){
    	std::stringstream tag;   
        tag << l;
        baseLayers[l].save(filename+"_baseLayer_"+tag.str());
    }  
}
void RNN::load(std::string filename){
    
    for (int l=0;l<numRecurrLayers;l++){
    	std::stringstream tag;
        tag << l;
        recurrLayers[l].load(filename+"_recurrLayer_"+tag.str());
    }
	for (int l=0;l<numBaseLayers;l++){
    	std::stringstream tag;   
        tag << l;
        baseLayers[l].load(filename+"_baseLayer_"+tag.str());
    }
}


// numerical gradient checking
// run this before the LSTM training, basically, need to use old weights to generate
// numerical gradients, and run the LSTM training by one iteration, forward() and 
// backward() to generate the model gradients which are compared to the numerical ones.
void RNN::calNumericGrad(){
    
    arma::mat delta;
    int dim1 = recurrLayers[0].outputDim;
    int dim2 = recurrLayers[0].inputOneDim;
    double eps = 1e-9;

    arma::mat dW(dim1, dim2, arma::fill::zeros);

    double temp_left, temp_right;
    double error;
    
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            (recurrLayers[0].W_one)->at(i, j) += eps;
            this->forward();
            error = 0.0;
            //           outputY->transform([](double val){return log(val);});
            for (int k = 0; k < trainingY->n_cols; k++) {
                delta = *(baseLayers[numBaseLayers - 1].outputMem[k]) - trainingY->col(k);
                error += arma::as_scalar(delta.st() * delta); 
            }
            error *= 0.5;
            temp_left = error;
            (recurrLayers[0].W_one)->at(i, j) -= 2.0 * eps;
            this->forward();
            //           outputY->transform([](double val){return log(val);});
            error = 0.0;
            for (int k = 0; k < trainingY->n_cols; k++) {
                delta = *(baseLayers[numBaseLayers - 1].outputMem[k]) - trainingY->col(k);
                error += arma::as_scalar(delta.st() * delta); 
            }
            error *= 0.5;
            temp_right = error;
            (recurrLayers[0].W_one)->at(i, j) += eps; // add back the change of the weights
            dW(i, j) = (temp_left - temp_right) / 2.0 / eps;
            
        }
    }
    dW.save("numGrad_recurrLayer.dat", arma::raw_ascii);
   
}

