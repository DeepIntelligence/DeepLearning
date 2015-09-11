#include "ElementMultiAddLayer.h"

using namespace NeuralNet;

ElementMultiAddLayer::ElementMultiAddLayer(){   
    grad_W_one = std::make_shared<arma::mat>();
    grad_W_two = std::make_shared<arma::mat>();
    delta_outOne = std::make_shared<arma::mat>();
    delta_outTwo = std::make_shared<arma::mat>();
    output = std::make_shared<arma::mat>();
}

void ElementMultiAddLayer::activateUp(){
    *output = (*W_one) % (*inputOne) + (*W_two) % (*inputTwo);
}


void ElementMultiAddLayer::calGrad(std::shared_ptr<arma::mat> delta_in){

    *grad_W_one = (*inputOne);
    *grad_W_two = (*inputTwo);

    (*delta_outOne) = *W_one;
    (*delta_outTwo) = *W_two;
}

void ElementMultiAddLayer::calGrad(std::shared_ptr<arma::mat> delta_in, int t){

    grad_W_one = inputOneMem[t];
    grad_W_two = inputTwoMem[t];

    delta_outOne = W_one_mem[t];
    delta_outTwo = W_two_mem[t];
}

