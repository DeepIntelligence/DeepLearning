#include "AutoEncoderLayer.h"
using namespace NeuralNet;
// constructor function
AutoEncoderLayer::AutoEncoderLayer(std::shared_ptr<arma::mat> trainingInputX,
                                   int outputLayerNum, TrainingPara trainingPara0):
    inputX(trainingInputX),outputDim(outputLayerNum),
    inputDim(trainingInputX->n_cols),
    trainingPara(trainingPara0) {
    std::shared_ptr<arma::mat> noise(new arma::mat(trainingInputX->n_rows,trainingInputX->n_cols));
    // add noise to each training sample
//   noise->randn(trainingInputX->n_rows,trainingInputX->n_cols);
    (*noise) += arma::randn(trainingInputX->n_rows,trainingInputX->n_cols);
    inputX_noise = noise;
    
    int numInstance = inputX -> n_rows;    
    initializeParameters();
}

// train function
void AutoEncoderLayer::pretrain() {



    arma::mat weightGrad(outputDim, inputDim);
    arma::vec biasGrad(outputDim);

    int iter = 0;
    double errorTotal;


    // matrix or vector form to deduct weight update by stochastic gradient descent
    arma::vec outputHidden(outputDim); // h
    arma::vec inLayerOut(inputDim); //

    arma::vec sumInToHidden (outputDim); // linear weighted sum of inputs to
    //  unit j in hidden layer
    arma::vec tempDelta(inputDim);
    int numInstance = inputX_noise->n_rows;
    while (iter < trainingPara.NEpoch) {
        errorTotal = 0.0;
        // m training instances
        std::cout << "iter: " << iter << std::endl;
        for (int i = 0; i < numInstance; i++) {

            // calculate the weight gradient firstly
            // "%" arma object element-wise multiplication
            //    W.print("W:");
            //    inputX_noise->row(i).st().print("X:");
            //    B.print("B:");
            outputHidden = W * ((inputX_noise->row(i)).st()) + B;
            activateFunc(AutoEncoderLayer::SIGMOID, outputHidden);
            inLayerOut = W.st() * outputHidden+ B_reconstruct;
            activateFunc(AutoEncoderLayer::SIGMOID,inLayerOut);
            arma::mat errorMat = (inLayerOut - inputX->row(i).st());
            tempDelta = errorMat % inLayerOut % (1-inLayerOut);

            weightGrad = outputHidden * tempDelta.st() + (((tempDelta % inputX_noise->row(i).st())*
                         (outputHidden % (1 - outputHidden)).st()) % W.st()).st();
           

            // calculate the B1 bias gradient secondly
            //biasGrad = (Wold.st() * tempDelta) % outputHidden % (1- outputHidden);
            biasGrad = (W * tempDelta);
            
            W -= trainingPara.alpha*weightGrad;

            B = B - trainingPara.alpha * biasGrad;

            errorTotal += arma::as_scalar(errorMat.st() * errorMat);
        }

        iter++;
        std::cout << "errorTotal " << errorTotal << std::endl;
    }
    W.save("finalWeight.dat",arma::raw_ascii);
}

void AutoEncoderLayer::initializeParameters() {
    W = (arma::randu(outputDim,inputDim)-0.5) * 4*sqrt(6.0/(inputDim+outputDim));
    B = (arma::randu(outputDim)-0.5)* 4*sqrt(6.0/(inputDim+outputDim));
    B_reconstruct = (arma::randu(inputDim)-0.5) * 4*sqrt(6.0/(inputDim+outputDim));
}

void AutoEncoderLayer::activateFunc(ActivationType actType, arma::mat &p) {
    _actType = actType;
    switch (actType) {
    case SIGMOID:
        p.transform([](double val) {
            return 1.0/(1.0+exp(-val));
        });
        break;
    case TANH:
        break;
    case CUBIC:
        break;
    case LINEAR:
        break;
    }
}

bool AutoEncoderLayer::converge(const arma::mat wGrad) {

    return norm(wGrad,2) < trainingPara.eps;
}
