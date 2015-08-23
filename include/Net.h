#pragma once
#include "common.h"

namespace NeuralNet{
class Net {
public:
    virtual ~Net() { }
    virtual void applyUpdates(std::vector<std::shared_ptr<arma::mat>>) = 0;
    virtual std::vector<std::shared_ptr<arma::mat>> netGradients() { return netGradVector;}
    virtual void setTrainingSamples(std::shared_ptr<arma::mat> X, std::shared_ptr<arma::mat> Y){ trainingX = X; trainingY = Y;}
	virtual void calGradient() = 0;
    virtual double getLoss() = 0;
    virtual void save(std::string filename) = 0;
    virtual void load(std::string filename) = 0;
    virtual void forward() = 0;
    virtual std::shared_ptr<arma::mat> netOutput() = 0;
    // the following are RNN specific
    virtual arma::mat forwardInTime(std::shared_ptr<arma::mat> x){}
    virtual std::shared_ptr<arma::mat> netOutputAtTime(int time){return 0;}
    virtual void zeroTime(){}
    virtual void updateInternalState(){}
    virtual void resetWeight(){}
protected:
	DeepLearning::NeuralNetParameter neuralNetPara;
	std::vector<std::shared_ptr<arma::mat>> netGradVector;
	std::shared_ptr<arma::mat> trainingX, trainingY;
};

}
