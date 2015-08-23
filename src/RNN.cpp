#include "RNN.h"


using namespace NeuralNet;
using namespace DeepLearning;

RNN::RNN(NeuralNetParameter neuralNetPara0){

    neuralNetPara = neuralNetPara0;
    // at beginning, we assume all the hidden layers have the same size,
    numHiddenLayers = neuralNetPara.rnnstruct().numhiddenlayers();
    hiddenLayerInputDim = neuralNetPara.rnnstruct().hiddenlayeroutputdim();
    hiddenLayerOutputDim = neuralNetPara.rnnstruct().hiddenlayeroutputdim();
    rnnInputDim = neuralNetPara.rnnstruct().inputdim();
    rnnOutputDim = neuralNetPara.layerstruct(0).outputdim();
    
    for (int i = 0; i < numHiddenLayers; i++){
        int inputOneDim, inputTwoDim;
		int outputDim;
		NeuralNetInitializerParameter  w_one_init, w_two_init, b_init;
        w_one_init = neuralNetPara.rnnstruct().init_w_one();
        w_two_init = neuralNetPara.rnnstruct().init_w_two();
        b_init = neuralNetPara.rnnstruct().init_b();       
        // i=0 is the first hidden layer with input consisting 
          // of data input and last time hidden
        if (i == 0) {           
            // inputDim is the unrolled LSTM input for various layers, the same to outputDim
            inputOneDim = hiddenLayerOutputDim;
            inputTwoDim = rnnInputDim;
            outputDim = hiddenLayerOutputDim;           
          // of hidden output from lower layer at the same time and
           // hidden output from same layer but at previous time
       } else if(i <= numHiddenLayers-1){
              
            // inputDim is the unrolled LSTM input for various layers, the same to outputDim
            inputOneDim = hiddenLayerOutputDim;
			inputTwoDim = hiddenLayerOutputDim;
            outputDim = hiddenLayerOutputDim;            
       }
        
		hiddenLayers.push_back(RecurrLayer(inputOneDim, inputTwoDim, outputDim, GetActivationType(neuralNetPara.rnnstruct().activationtype()),
		InitializerBuilder::GetInitializer(w_one_init), InitializerBuilder::GetInitializer(w_two_init), 
		InitializerBuilder::GetInitializer(b_init)));     
	}
        NeuralNetInitializerParameter  w_init, b_init;
        w_init = neuralNetPara.layerstruct(0).init_w();
        b_init = neuralNetPara.layerstruct(0).init_b();
 
		netOutputLayer = std::make_shared<BaseLayer>(neuralNetPara.layerstruct(0).inputdim(),
        neuralNetPara.layerstruct(0).outputdim(), GetActivationType(neuralNetPara.layerstruct(0).activationtype()),
        InitializerBuilder::GetInitializer(w_init), InitializerBuilder::GetInitializer(b_init));

        fillNetGradVector();
}

void RNN::resetWeight(){
	for (int l = 0; l < numHiddenLayers; l++) {
        hiddenLayers[l].initializeWeight();
    }
    netOutputLayer->initializeWeight();
}
void RNN::setTime(int t){ this->time = t;}
int RNN::getTime(){return this->time;}
void RNN::zeroTime(){this->time = 0;}

arma::mat RNN::forwardInTime(std::shared_ptr<arma::mat> input) {
    std::shared_ptr<arma::mat> commonInput(new arma::mat);
    if (this->time == 0) {
        for (int l = 0; l < numHiddenLayers; l++) {
            (hiddenLayers[l].getPrevOutput())->zeros(hiddenLayerOutputDim, 1);
            hiddenLayers[l].inputOneMem.clear();
            hiddenLayers[l].inputTwoMem.clear();
            hiddenLayers[l].outputMem.clear();
        }
        netOutputLayer->inputMem.clear();
        netOutputLayer->outputMem.clear();
    }
    for (int l = 0; l < numHiddenLayers; l++) {
    	hiddenLayers[l].inputOne = std::shared_ptr<arma::mat>(new arma::mat(*(hiddenLayers[l].getPrevOutput())));
  
        if (l == 0) {
			hiddenLayers[l].inputTwo = std::shared_ptr<arma::mat>(new arma::mat(*input));
        } else {
			hiddenLayers[l].inputTwo = std::shared_ptr<arma::mat>(new arma::mat(*(hiddenLayers[l - 1].output)));
        }
        hiddenLayers[l].activateUp();
#if 0
        hiddenLayers[l].W_one->print("W_one");
        hiddenLayers[l].W_two->print("W_two");
        hiddenLayers[l].B->print("B");
        hiddenLayers[l].inputOne->print("input one");
        hiddenLayers[l].inputTwo->print("input two");
        hiddenLayers[l].output->print("output");
#endif        
    }
    netOutputLayer->input = hiddenLayers[numHiddenLayers-1].output;
    netOutputLayer->activateUp();

    return *(netOutputLayer->output);
}
void RNN::saveLayerInputOutput(){
    for (int l = 0; l < numHiddenLayers; l++) {
        hiddenLayers[l].saveInputMemory();
        hiddenLayers[l].saveOutputMemory();
    }
    netOutputLayer->saveInputMemory();
    netOutputLayer->saveOutputMemory();
        
}

void RNN::updateInternalState(){
    for (int l = 0; l < numHiddenLayers; l++) {
    	hiddenLayers[l].savePrevOutput();
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

    for (int l = 0; l < numHiddenLayers; l++) {
//        hiddenLayer_upstream_deltaOut[l].zeros(hiddenLayerOutputDim, 1);
        hiddenLayers[l].clearAccuGrad();
    }
    netOutputLayer->clearAccuGrad();
      
    int T = trainingY->n_cols;
    for (int t = T - 1; t >= 0; t--){
        *delta = *(netOutputLayer->outputMem[t]) - trainingY->col(t);
        netOutputLayer->accumulateGrad(delta, t);
        for (int l = numHiddenLayers - 1; l >= 0; l--){
            // delta error from the same time, propagate from upper layer to lower layer 
            if (l == numHiddenLayers - 1){ // the top most layer from target - network's output
                *delta = *(netOutputLayer->delta_out); 
            }else{ 
                *delta = *(hiddenLayers[l+1].delta_outTwo);                
            }   
            
            if (t < T - 1) {
               *delta += *(hiddenLayers[l].getPrevDeltaOutOne());
            }              
            // so far, the generated delta error is for the output h of each layer at each time
            hiddenLayers[l].accumulateGrad(delta, t);
            hiddenLayers[l].savePrevDeltaOutOne();
//            hiddenLayer_upstream_deltaOut[l] = *(hiddenLayers[l].delta_outOne);
        }
	}  
}

void RNN::fillNetGradVector(){
    for (int i = 0; i < this->numHiddenLayers; i++){
        netGradVector.push_back(this->hiddenLayers[i].grad_W_one_accu);
        netGradVector.push_back(this->hiddenLayers[i].grad_W_two_accu);
        netGradVector.push_back(this->hiddenLayers[i].grad_B_accu);
    }
    netGradVector.push_back((this->netOutputLayer)->grad_W_accu);
    netGradVector.push_back((this->netOutputLayer)->grad_B_accu);
}

void RNN::applyUpdates(std::vector<std::shared_ptr<arma::mat>> inGradVector){
    for (int i = 0; i < this->numHiddenLayers; i++){
        *(hiddenLayers[i].W_one) -= *(inGradVector[3*i]);
        *(hiddenLayers[i].W_two) -= *(inGradVector[3*i+1]);
        *(hiddenLayers[i].B) -= *(inGradVector[3*i+2]);
    }
    *(netOutputLayer->W) -= *(inGradVector[3*this->numHiddenLayers]);
    *(netOutputLayer->B) -= *(inGradVector[3*this->numHiddenLayers+1]);    
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
        delta = *((netOutputLayer->outputMem)[k]) - trainingY->col(k);
        error += arma::as_scalar(delta.st() * delta); 
    }
    return error;
}

std::shared_ptr<arma::mat> RNN::netOutput(){
    netOutput_ = std::make_shared<arma::mat>(this->rnnOutputDim, trainingX->n_cols);
    for (int k = 0; k < trainingX->n_cols; k++) {
        double* ptr = netOutput_->colptr(k);
        for (int i = 0; i < rnnOutputDim; i++){
            *(ptr+i) = (netOutputLayer->outputMem[k])->at(i);
        }
    }
    return netOutput_;
}

std::shared_ptr<arma::mat> RNN::netOutputAtTime(int time){
    return netOutputLayer->outputMem[time];
}

void RNN::save(std::string filename){
    std::stringstream tag;   
    for (int l=0;l<numHiddenLayers;l++){
        tag << l;
        hiddenLayers[l].save(filename+"_hiddenLayer_"+tag.str());
    }
    this->netOutputLayer->save(filename+"netOutputLayer");
}
void RNN::load(std::string filename){
    std::stringstream tag;
    for (int l=0;l<numHiddenLayers;l++){
        tag << l;
        hiddenLayers[l].load(filename+"_hiddenLayer_"+tag.str());
    }
    this->netOutputLayer->load(filename+"netOutputLayer");
}


// numerical gradient checking
// run this before the LSTM training, basically, need to use old weights to generate
// numerical gradients, and run the LSTM training by one iteration, forward() and 
// backward() to generate the model gradients which are compared to the numerical ones.
void RNN::calNumericGrad(){
    
    arma::mat delta;
    int dim1 = hiddenLayers[0].outputDim;
    int dim2 = hiddenLayers[0].inputOneDim;
    double eps = 1e-9;

    arma::mat dW(dim1, dim2, arma::fill::zeros);

    double temp_left, temp_right;
    double error;
    
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            (hiddenLayers[0].W_one)->at(i, j) += eps;
            this->forward();
            error = 0.0;
            //           outputY->transform([](double val){return log(val);});
            for (int k = 0; k < trainingY->n_cols; k++) {
                delta = *(netOutputLayer->outputMem[k]) - trainingY->col(k);
                error += arma::as_scalar(delta.st() * delta); 
            }
            error *= 0.5;
            temp_left = error;
            (hiddenLayers[0].W_one)->at(i, j) -= 2.0 * eps;
            this->forward();
            //           outputY->transform([](double val){return log(val);});
            error = 0.0;
            for (int k = 0; k < trainingY->n_cols; k++) {
                delta = *(netOutputLayer->outputMem[k]) - trainingY->col(k);
                error += arma::as_scalar(delta.st() * delta); 
            }
            error *= 0.5;
            temp_right = error;
            (hiddenLayers[0].W_one)->at(i, j) += eps; // add back the change of the weights
            dW(i, j) = (temp_left - temp_right) / 2.0 / eps;
            
        }
    }
    dW.save("numGrad_hiddenLayer.dat", arma::raw_ascii);
   
}

