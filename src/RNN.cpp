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
    rnnOutputDim = neuralNetPara.rnnstruct().outputdim(); // this parameter is not used within sofar code
     
    for (int i = 0; i < numHiddenLayers; i++){
        int inputOneDim, inputTwoDim;
		int outputDim;
		NeuralNetInitializerParameter  w_one_init, w_two_init, b_init;
        w_one_init = neuralNetPara.layerstruct(i).init_w_one();
        w_two_init = neuralNetPara.layerstruct(i).init_w_two();
        b_init = neuralNetPara.layerstruct(i).init_b();
        // i=0 is the first hidden layer with input consisting 
          // of data input and last time hidden
        if (i == 0) {
            
            // inputDim is the unrolled LSTM input for various layers, the same to outputDim
            inputOneDim = rnnInputDim;
			inputTwoDim = hiddenLayerOutputDim;
            outputDim = hiddenLayerOutputDim;
            

            
          // of hidden output from lower layer at the same time and
           // hidden output from same layer but at previous time
       } else if(i <= numHiddenLayers-1){
              
            // inputDim is the unrolled LSTM input for various layers, the same to outputDim
            inputOneDim = hiddenLayerOutputDim;
			inputTwoDim = hiddenLayerOutputDim;
            outputDim = hiddenLayerOutputDim;
            
        }
        
			switch (neuralNetPara.rnnstruct().activationtype()) {
				case RNNStructParameter_ActivationType_sigmoid:
					hiddenLayers.push_back(MultiAddLayer(inputOneDim, inputTwoDim, outputDim, sigmoid,
					InitializerBuilder::GetInitializer(w_one_init), InitializerBuilder::GetInitializer(w_two_init), 
					InitializerBuilder::GetInitializer(b_init)));     
        			break;
        		case RNNStructParameter_ActivationType_tanh:
					hiddenLayers.push_back(MultiAddLayer(inputOneDim, inputTwoDim, outputDim, tanh,
					InitializerBuilder::GetInitializer(w_one_init), InitializerBuilder::GetInitializer(w_two_init), 
					InitializerBuilder::GetInitializer(b_init)));     
        			break;
        		case RNNStructParameter_ActivationType_softmax:
					hiddenLayers.push_back(MultiAddLayer(inputOneDim, inputTwoDim, outputDim, softmax,
					InitializerBuilder::GetInitializer(w_one_init), InitializerBuilder::GetInitializer(w_two_init), 
					InitializerBuilder::GetInitializer(b_init)));     
        			break;
        		case RNNStructParameter_ActivationType_ReLU:
					hiddenLayers.push_back(MultiAddLayer(inputOneDim, inputTwoDim, outputDim, ReLU,
					InitializerBuilder::GetInitializer(w_one_init), InitializerBuilder::GetInitializer(w_two_init), 
					InitializerBuilder::GetInitializer(b_init)));     
        			break;
        		case RNNStructParameter_ActivationType_linear:
					hiddenLayers.push_back(MultiAddLayer(inputOneDim, inputTwoDim, outputDim, linear,
					InitializerBuilder::GetInitializer(w_one_init), InitializerBuilder::GetInitializer(w_two_init), 
					InitializerBuilder::GetInitializer(b_init)));     
        			break;
        		default:
        			break;
			}
		}

        NeuralNetInitializerParameter  w_init, b_init;
        w_init = neuralNetPara.layerstruct(0).init_w();
        b_init = neuralNetPara.layerstruct(0).init_b();
 
        switch (neuralNetPara.layerstruct(0).activationtype()) {
            case LayerStructParameter_ActivationType_sigmoid:
                netOutputLayer = std::make_shared<BaseLayer>(neuralNetPara.layerstruct(0).inputdim(),
                        neuralNetPara.layerstruct(0).outputdim(), sigmoid,
                        InitializerBuilder::GetInitializer(w_init), InitializerBuilder::GetInitializer(b_init));
                break;
            case LayerStructParameter_ActivationType_tanh:
                netOutputLayer = std::make_shared<BaseLayer>(neuralNetPara.layerstruct(0).inputdim(),
                        neuralNetPara.layerstruct(0).outputdim(), tanh,
                        InitializerBuilder::GetInitializer(w_init), InitializerBuilder::GetInitializer(b_init));
                break;
            case LayerStructParameter_ActivationType_softmax:
                netOutputLayer = std::make_shared<BaseLayer>(neuralNetPara.layerstruct(0).inputdim(),
                        neuralNetPara.layerstruct(0).outputdim(), softmax,
                        InitializerBuilder::GetInitializer(w_init), InitializerBuilder::GetInitializer(b_init));
                break;
            case LayerStructParameter_ActivationType_linear:
                netOutputLayer = std::make_shared<BaseLayer>(neuralNetPara.layerstruct(0).inputdim(),
                        neuralNetPara.layerstruct(0).outputdim(), linear,
                        InitializerBuilder::GetInitializer(w_init), InitializerBuilder::GetInitializer(b_init));
                break;
            case LayerStructParameter_ActivationType_ReLU:
                netOutputLayer = std::make_shared<BaseLayer>(neuralNetPara.layerstruct(0).inputdim(),
                        neuralNetPara.layerstruct(0).outputdim(), ReLU,
                        InitializerBuilder::GetInitializer(w_init), InitializerBuilder::GetInitializer(b_init));
                break;
            default:break;
        }
    
        fillNetGradVector();
         for (int l = 0; l < numHiddenLayers; l++) {
            outputLayers_prev_output.push_back(std::shared_ptr<arma::mat>(new arma::mat));
        }
}

void RNN::setTime(int t){ this->time = t;}
int RNN::getTime(){return this->time;}

arma::mat RNN::forwardInTime(std::shared_ptr<arma::mat> input) {
    std::shared_ptr<arma::mat> commonInput(new arma::mat);
    if (this->time == 0) {
        for (int l = 0; l < numHiddenLayers; l++) {
            (outputLayers_prev_output[l])->zeros(hiddenLayerOutputDim, 1);
            hiddenLayers[l].inputOneMem.clear();
			hiddenLayers[l].inputTwoMem.clear();
            hiddenLayers[l].outputMem.clear();
        }
        netOutputLayer->inputMem.clear();
        netOutputLayer->outputMem.clear();
    }
    for (int l = 0; l < numHiddenLayers; l++) {
        // concatenate to a large vector            
        if (l == 0) {
			hiddenLayers[l].inputOne = outputLayers_prev_output[l];
			hiddenLayers[l].inputTwo = input;
        } else {
			hiddenLayers[l].inputOne = outputLayers_prev_output[l];
			hiddenLayers[l].inputTwo = hiddenLayers[l - 1].output;
        }

        hiddenLayers[l].activateUp();
 
        if (l == numHiddenLayers - 1) {

            netOutputLayer->input = hiddenLayers[l].output;
            netOutputLayer->activateUp();
        }
    }
    return *(netOutputLayer->output);
}
void RNN::saveLayerInputOutput(){
    for (int l = 0; l < numHiddenLayers; l++) {
        hiddenLayers[l].saveInputMemory();
        hiddenLayers[l].saveOutputMemory();
 
        if (l == numHiddenLayers - 1) {
            netOutputLayer->saveInputMemory();
            netOutputLayer->saveOutputMemory();
        }
    }
}



void RNN::updateInternalState(){
    for (int l = 0; l < numHiddenLayers; l++) {        
        *(outputLayers_prev_output[l]) = *(hiddenLayers[l].output);
    }
}

void RNN::forward() {
    int T = trainingX->n_cols; 
    for (this->time = 0; this->time < T; (this->time)++){
        std::shared_ptr<arma::mat> input = std::make_shared<arma::mat>(trainingX->col(this->time));
        this->forwardInTime(input);
        this->saveLayerInputOutput();
        this->updateInternalState();
    }
}
 
void RNN::backward() {
   
    arma::mat hiddenLayer_upstream_deltaOut[numHiddenLayers]; 
    
    std::shared_ptr<arma::mat> delta, delta_upstream; // temporal delta is a vector
    delta = std::make_shared<arma::mat>();

    for (int l = 0; l < numHiddenLayers; l++) {
        hiddenLayer_upstream_deltaOut[l].zeros(hiddenLayerOutputDim, 1);
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
               *delta += hiddenLayer_upstream_deltaOut[l];
            }              
            // so far, the generated delta error is for the output h of each layer at each time
            hiddenLayers[l].accumulateGrad(delta, t);
            hiddenLayer_upstream_deltaOut[l] = *(hiddenLayers[l].delta_outOne);
        }
	}
        // save this accumulated gradients for comparing with numerical gradients
  //      if (l==0){ // save this l layer
  //         hiddenLayers[l].grad_W_accu.save("hiddenLayer0_Grad.dat", arma::raw_ascii);
  //      }   
}

void RNN::test(){

}

void RNN::setTrainingSamples(std::shared_ptr<arma::mat> X, std::shared_ptr<arma::mat> Y){
    this->trainingX = X;
    this->trainingY = Y;
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

std::vector<std::shared_ptr<arma::mat>> RNN::netGradients(){
    return this->netGradVector;
}
double RNN::getLoss(){
//  for calcuating the total error
    double error = 0.0;
    arma::mat delta;
    //           outputY->transform([](double val){return log(val);});
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
    char tag[10];    
    for (int l=0;l<numHiddenLayers;l++){
        sprintf(tag,"%d",l);
        hiddenLayers[l].save(filename+"_hiddenLayer_"+(std::string)tag);
    }
    this->netOutputLayer->save(filename+"netOutputLayer");
}
void RNN::load(std::string filename){
    char tag[10];    
    for (int l=0;l<numHiddenLayers;l++){
        sprintf(tag,"%d",l);
        hiddenLayers[l].load(filename+"_hiddenLayer_"+(std::string)tag);
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
    double eps = 1e-5;

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

