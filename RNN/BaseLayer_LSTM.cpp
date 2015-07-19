#include "BaseLayer_LSTM.h"

using namespace NeuralNet;

// save inputs at all time points during the LSTM forward pass
void BaseLayer_LSTM::saveInputMemory() {
    inputMem.push_back(input);
}

// extract out the specific input at time point t during backpropagation
// to calculate the gradient
std::shared_ptr<arma::mat> BaseLayer_LSTM::getInputMemory(int t) {
    return inputMem[t];
}

void BaseLayer_LSTM::calGrad(std::shared_ptr<arma::mat> delta_in, int timePoint) {
    //for delta: each column is the delta of a sample
    arma::mat deriv;
    arma::mat delta;
    delta_out = std::make_shared<arma::mat>(inputDim, delta_in->n_cols);
    if (actType == softmax) {
        deriv.ones(output->n_rows, output->n_cols);
    } else if (actType == sigmoid) {
        deriv = (1 - (*output)) % (*output);
    } else if (actType == tanh) {
        deriv = (1 - (*output) % (*output));
    } else if (actType == linear) {
        deriv.ones(output->n_rows, output->n_cols);
    }
    delta = (*delta_in) % deriv;
    grad_B = arma::sum(delta, 1);
    
    std::shared_ptr<arma::mat> tempInput = getInputMemory(timePoint);
    grad_W = delta * (*tempInput).st();
#if 0
    if (dropOutFlag) {
        // for each column
        delta = delta % dropOutMat;
    }
#endif    
    (*delta_out) = W.st() * (delta);
}

void BaseLayer_LSTM::accumulateGrad(std::shared_ptr<arma::mat> delta_in, int t) {
    calGrad(delta_in, t);
    grad_B_accu += grad_B;
    grad_W_accu += grad_W;
}