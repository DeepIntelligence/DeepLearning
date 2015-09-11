#include "MultiAddLayer.h"

using namespace NeuralNet;

MultiAddLayer::MultiAddLayer(int in1, int in2, int out, ActivationType actType0, std::shared_ptr<Initializer> init_W_one, std::shared_ptr<Initializer> init_W_two,
        std::shared_ptr<Initializer> init_B):
        Layer_binaryOp(in1,in2,out),actType(actType0),
        initializer_W_one(init_W_one), initializer_W_two(init_W_two),
        initializer_B(init_B){   
    initializeWeight();
    grad_W_one = std::make_shared<arma::mat>();
    grad_W_two = std::make_shared<arma::mat>();
    grad_B = std::make_shared<arma::mat>();
    grad_W_one_accu = std::make_shared<arma::mat>(outputDim,inputOneDim, arma::fill::zeros);
    grad_W_two_accu = std::make_shared<arma::mat>(outputDim,inputTwoDim, arma::fill::zeros);
    grad_B_accu = std::make_shared<arma::mat>(outputDim,1, arma::fill::zeros);
    delta_outOne = std::make_shared<arma::mat>();
    delta_outTwo = std::make_shared<arma::mat>();
    output = std::make_shared<arma::mat>();

}

void MultiAddLayer::activateUp(){
    *output = (*W_one) * (*inputOne) + (*W_two) * (*inputTwo);
    for (int i = 0; i < output->n_cols; i++) output->col(i) += *B;
    ApplyActivation(output, actType);
}

void MultiAddLayer::initializeWeight(){

    W_one = std::make_shared<arma::mat>(outputDim, inputOneDim);
    W_two = std::make_shared<arma::mat>(outputDim, inputTwoDim);
    B = std::make_shared<arma::mat>(outputDim, 1);

    if (initializer_W_one == nullptr || initializer_W_two == nullptr ||initializer_B == nullptr) {
        std::cerr << "initializer is null!" << std::endl;
        exit(1);
    } else {
        initializer_W_one->applyInitialization(W_one);
        initializer_W_two->applyInitialization(W_two);
        initializer_B->applyInitialization(B);
    }
}

void MultiAddLayer::calGrad(std::shared_ptr<arma::mat> delta_in){
    //for delta: each column is the delta of a sample
    std::shared_ptr<arma::mat> deriv(new arma::mat);
    GetActivationGradient(output, deriv, this->actType);
    arma::mat delta;
 
    delta = (*delta_in) % (*deriv);
    *grad_B = arma::sum(delta,1);
    *grad_W_one = delta * (*inputOne).st();
    *grad_W_two = delta * (*inputTwo).st();

    (*delta_outOne) = W_one->st() * (delta);
    (*delta_outTwo) = W_two->st() * (delta);
}

void MultiAddLayer::calGrad(std::shared_ptr<arma::mat> delta_in, int t){
    std::shared_ptr<arma::mat> deriv(new arma::mat);
    GetActivationGradient(outputMem[t], deriv, this->actType);
    arma::mat delta;
 
    delta = (*delta_in) % (*deriv);
    *grad_B = arma::sum(delta,1);
    *grad_W_one = delta * (*inputOneMem[t]).st();
    *grad_W_two = delta * (*inputTwoMem[t]).st();

    (*delta_outOne) = W_one->st() * (delta);
    (*delta_outTwo) = W_two->st() * (delta);
}

void MultiAddLayer::save(std::string filename) {
    W_one->save(filename+"_W_one.dat",arma::raw_ascii);
    W_two->save(filename+"_W_two.dat",arma::raw_ascii);
    B->save(filename+"_B.dat",arma::raw_ascii);
}
void MultiAddLayer::load(std::string filename) {
    W_one->load(filename+"_W_one.dat",arma::raw_ascii);
    W_two->load(filename+"_W_two.dat",arma::raw_ascii);
    B->load(filename+"_B.dat",arma::raw_ascii);
}

void MultiAddLayer::accumulateGrad(std::shared_ptr<arma::mat> delta_in, int t) {
    calGrad(delta_in, t);
    *grad_B_accu += *grad_B;
    *grad_W_one_accu += *grad_W_one;
    *grad_W_two_accu += *grad_W_two;
}

void MultiAddLayer::clearAccuGrad(){
    (*grad_B_accu).zeros();
    (*grad_W_one_accu).zeros();
    (*grad_W_two_accu).zeros();
}