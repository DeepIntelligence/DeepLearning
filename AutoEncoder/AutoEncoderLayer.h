#pragma once
#include <armadillo>
#include "../BaseLayer/BaseLayer.h"

namespace NeuralNet{

struct TrainingPara {
//  enum Method{gradDescent,stochGradDescent};
    TrainingPara(double eps0=1e-6, int NEpoch0 = 20,
                 int miniBatchSize0 = 10,
                 double alpha0 = 0.01):
        eps(eps0),NEpoch(NEpoch0),
        miniBatchSize(miniBatchSize0),
        alpha(alpha0) {}


    double eps;
    int NEpoch;
    int miniBatchSize;
    double alpha;
//  Method method;
};
// Two layer AutoEncoder with tight weight W'=WT
// update weight by backpropagation x~ reconstruction
// stacked denoised AutoEncoder
// sparsed AuEncoder
class AutoEncoderLayer {

public:
    enum ActivationType {SIGMOID, TANH, CUBIC, LINEAR};
    AutoEncoderLayer(std::shared_ptr<arma::mat> trainingInputX,
                     int outputLayerNum, TrainingPara trainingPara0);
    void pretrain();
    void initializeParameters();
    bool converge(const arma::mat wUpdate);
    void activateFunc(ActivationType actType, arma::mat& p);


private:

    int outputDim;
    int inputDim;
    int alpha; // learning rate;
    int ntimes;
    ActivationType _actType;
    arma::mat W; // reconstructed weight of one layer
    arma::vec B, B_reconstruct; // bias for the two layers
    arma::mat output; // hidden layer nodes
    arma::vec inOut;
    std::shared_ptr<arma::mat> inputX; // training samples features
    //std::shared_ptr<arma::mat> outputY; // training samples features

    std::shared_ptr<arma::mat> inputX_noise;
//  std::shared_ptr<arma::vec> trainingY;// training samples labels

    double noiseLevel;
//  std::shared_ptr<arma::randn> noise;
    TrainingPara trainingPara;

};
}