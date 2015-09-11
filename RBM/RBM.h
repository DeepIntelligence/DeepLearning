#pragma once
#include <vector>
#include <string>
#include <memory>
#include <armadillo>
#include "../Utils/Util.h"

namespace NeuralNet{

class RBM {
public:
    struct PreTrainPara {

        PreTrainPara(double eps0=1e-6, int NEpoch0 = 500,
                     int miniBatchSize0 = 10, double alpha0 = 0.01,
                     double momentum0 = 0.9, int saveFreq0 = 50, 
                     double learningRateDecay0 = 1.0, bool dropOutFlag0 = false,
                     double dropOutRate0 = 0.3, double L2Decay0 = 0.0002):
            eps(eps0),NEpoch(NEpoch0), miniBatchSize(miniBatchSize0), 
            alpha(alpha0), momentum(momentum0), saveFrequency(saveFreq0),
            learningRateDecay(learningRateDecay0), dropOutFlag(dropOutFlag0),
            dropOutRate(dropOutRate0), L2Decay(L2Decay0){}
        double eps;
        int NEpoch;
        int miniBatchSize;
        double alpha;
        double momentum;
        int saveFrequency;
        double learningRateDecay;
        bool dropOutFlag;
        double dropOutRate; 
        double L2Decay;
        void print() const;
    };


    RBM(int visibleDim, int hiddenDim, RBM::PreTrainPara preTrainPara0);
    RBM(int visibleDim, int hiddenDim, std::shared_ptr<arma::umat> trainingX0, RBM::PreTrainPara preTrainPara0);
    void train();
    void saveTrainResult(std::string filename);
    void loadTrainResult(std::string filename);
    void initializeWeight();
    void propUp(std::shared_ptr<arma::umat>);
    void reconstructVisible();
    void reconstructHiddenProb();
    double calReconstructError(std::shared_ptr<arma::umat> inputX);
    double calEnergy(std::shared_ptr<arma::umat> inputX) const;
    void TestViaReconstruct(std::shared_ptr<arma::mat> testDataX);
    int inputDim;
    int outputDim;
    int numInstance;
    Random_Bernoulli<unsigned long long> *randomGen;
    std::shared_ptr<arma::mat> inputX, W , outputY, H_reconstructProb, grad_W, grad_W_old;
    std::shared_ptr<arma::umat > H,V, V_reconstruct;
    std::shared_ptr<arma::vec> A, B, grad_B, grad_B_old, grad_A, grad_A_old;
    RBM::PreTrainPara trainingPara;

};

}