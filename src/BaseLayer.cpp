#include <cmath>
#include <random>
#include <assert.h>
#include "BaseLayer.h"

using namespace NeuralNet;

BaseLayer::BaseLayer(int inputDim0, int outputDim0, ActivationType actType0, 
        std::shared_ptr<Initializer> init_W, std::shared_ptr<Initializer> init_B,
		bool dropout, double dropr): Layer_unitaryOp(inputDim0, outputDim0),
    actType(actType0), initializer_W(init_W), initializer_B(init_B), dropOutFlag(dropout), dropOutRate(dropr){
    
    
    initializeWeight();
    grad_W = std::make_shared<arma::mat>(outputDim,inputDim);
    grad_B = std::make_shared<arma::mat>(outputDim,1);
    grad_W_accu = std::make_shared<arma::mat>(outputDim,inputDim);
    grad_B_accu = std::make_shared<arma::mat>(outputDim,1);
    grad_W_accu->zeros();
    grad_B_accu->zeros();
    W_size = inputDim * outputDim;
    B_size = outputDim;
    totalSize = W_size + B_size;
    delta_out = std::make_shared<arma::mat>();
    output = std::make_shared<arma::mat>();
    
    if (dropOutFlag) {
        randomGen=new Random_Bernoulli<double>(dropOutRate);
    }
};

void BaseLayer::initializeWeight() {

	W = std::make_shared<arma::mat>(outputDim, inputDim);
	B = std::make_shared<arma::mat>(outputDim, 1);

	if (initializer_W == nullptr || initializer_B == nullptr) {
		std::cerr << "initializer is null!" << std::endl;
		exit(1);
	} else {
		initializer_W->applyInitialization(W);
		initializer_B->applyInitialization(B);
	}
}

void BaseLayer::save(std::string filename) {
    W->save(filename+"_W.dat",arma::raw_ascii);
    B->save(filename+"_B.dat",arma::raw_ascii);
}
void BaseLayer::load(std::string filename) {
    W->load(filename+"_W.dat",arma::raw_ascii);
    B->load(filename+"_B.dat",arma::raw_ascii);
}


void BaseLayer::accumulateGrad(std::shared_ptr<arma::mat> delta_in){
    calGrad(delta_in);
    *grad_B_accu += *grad_B;
    *grad_W_accu += *grad_W;
}

void BaseLayer::calGrad(std::shared_ptr<arma::mat> delta_in){
    //for delta: each column is the delta of a sample
    std::shared_ptr<arma::mat> deriv(new arma::mat);
	GetActivationGradient(output, deriv, this->actType);

    arma::mat delta;
 
    delta = (*delta_in) % (*deriv);
//    delta_in->print();
//    delta.print();
    *grad_B = arma::sum(delta,1);
    *grad_W = delta * (*input).st();

    if(dropOutFlag) {
        // for each column
       delta = delta % dropOutMat; 
    }
    (*delta_out) = W->st() * (delta);

}

void BaseLayer::activateUp(){
	assert((this->input!=NULL)&&"null ptr in the activateUp()");
	this->activateUp(this->input);
}
void BaseLayer::activateUp(std::shared_ptr<arma::mat> input0) {
    if(dropOutFlag){
//        BaseLayer::fill_Bernoulli(dropOutMat.memptr(),W_size);
    }
    input = input0;
    std::shared_ptr<arma::mat> &p=output;
    if( dropOutFlag) {
        *input = (*input) % dropOutMat;
    }
    (*output) = (*W) * (*input);
    for (int i = 0; i < input->n_cols; i++) p->col(i) += *B;
    ApplyActivation(output, this->actType);
}

// extract out the specific input at time point t during backpropagation
// to calculate the gradient
std::shared_ptr<arma::mat> BaseLayer::getInputMemory(int t) {
    return inputMem[t];
}


// extract out the specific output at time point t during backpropagation
// to calculate the gradient
std::shared_ptr<arma::mat> BaseLayer::getOutputMemory(int t){
    return outputMem[t];
}


void BaseLayer::calGrad(std::shared_ptr<arma::mat> delta_in, int timePoint) {
    //for delta: each column is the delta of a sample
    std::shared_ptr<arma::mat> deriv(new arma::mat);
    arma::mat delta;
    std::shared_ptr<arma::mat> tempOutput = getOutputMemory(timePoint);
    GetActivationGradient(tempOutput, deriv, this->actType);
    
    delta_out = std::make_shared<arma::mat>(inputDim, delta_in->n_cols);

    delta = (*delta_in) % (*deriv);
    *grad_B = arma::sum(delta, 1);
    
    std::shared_ptr<arma::mat> tempInput = getInputMemory(timePoint);
    *grad_W = delta * (*tempInput).st();
#if 0
    if (dropOutFlag) {
        // for each column
        delta = delta % dropOutMat;
    }
#endif    
    (*delta_out) = W->st() * (delta);
}

void BaseLayer::accumulateGrad(std::shared_ptr<arma::mat> delta_in, int t) {
    calGrad(delta_in, t);
    *grad_B_accu += *grad_B;
    *grad_W_accu += *grad_W;
}

void BaseLayer::clearAccuGrad(){
    (*grad_B_accu).zeros();
    (*grad_W_accu).zeros();
}


void BaseLayer::vectoriseGrad(std::shared_ptr<arma::vec> V){
    
    *V = arma::vectorise(*grad_W);
    V->resize(W_size + B_size);
    V->rows(W_size,W_size+B_size-1) = *grad_B;
}

void BaseLayer::vectoriseWeight(std::shared_ptr<arma::vec> V){
    
    *V = arma::vectorise(*W);
    V->resize(W_size + B_size);
    V->rows(W_size,W_size+B_size-1) =*B;
}


void BaseLayer::deVectoriseWeight(std::shared_ptr<arma::vec> V){
    
    *B =  V->rows(W_size,W_size+B_size-1);
    V->resize(W_size);
    *W = *V;
    W->reshape(outputDim, inputDim);
}
// vectorise grad is frequency used to pass out the gradient as a vector
void BaseLayer::vectoriseGrad(double *ptr, size_t offset){

    double *W_ptr = grad_W->memptr();
    double *B_ptr = grad_B->memptr();
    for (int i = 0; i < W_size; i++){
        *(ptr + offset) = *(W_ptr+i);
        offset++;
    }
    for (int i = 0; i < B_size; i++){
        *(ptr + offset) = *(B_ptr+i);
        offset++;
    }
}

void BaseLayer::vectoriseWeight(double *ptr, size_t offset){
    
    double *W_ptr = W->memptr();
    double *B_ptr = B->memptr();
    for (int i = 0; i < W_size; i++){
        *(ptr + offset) = *(W_ptr+i);
        offset++;
    }
    for (int i = 0; i < B_size; i++){
        *(ptr + offset) = *(B_ptr+i);
        offset++;
    }
}

// devectorise weight is frequency used to pass out the gradient as a vector
void BaseLayer::deVectoriseWeight(double *ptr, size_t offset){
    
    double *W_ptr = W->memptr();
    double *B_ptr = B->memptr();
    for (int i = 0; i < W_size; i++){
        *(W_ptr+i) = *(ptr + offset) ;
        offset++;
    }
    for (int i = 0; i < B_size; i++){
        *(B_ptr+i) = *(ptr + offset) ;
        offset++;
    }
}

void BaseLayer::fill_Bernoulli(double *p, int size){

    for (int i = 0; i < size; i++){
        if(randomGen->next()) *(p+i) = 1.0;
        else *(p+i) = 0.0;
    }
}


