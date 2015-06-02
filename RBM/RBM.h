#pragma once
#include <vector>
#include <string>
#include <memory>
#include <armadillo>



class RBM {
public:
    struct PreTrainPara {

        PreTrainPara(double eps0=1e-6, int NEpoch0 = 500,
                     int miniBatchSize0 = 10, double alpha0 = 0.1):
            eps(eps0),NEpoch(NEpoch0),
            miniBatchSize(miniBatchSize0), alpha(alpha0) {}

        double eps;
        int NEpoch;
        int miniBatchSize;
        double alpha;
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
    std::shared_ptr<arma::mat> inputX, W , outputY, H_reconstructProb;
    std::shared_ptr<arma::umat > H,V, V_reconstruct;
    std::shared_ptr<arma::vec> A, B;

    RBM::PreTrainPara trainingPara;

};
