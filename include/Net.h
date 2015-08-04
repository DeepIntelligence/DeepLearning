#pragma once
#include "common.h"

class Net {
public:
    virtual void applyUpdates(std::vector<std::shared_ptr<arma::mat>>) = 0;
    virtual std::vector<std::shared_ptr<arma::mat>> netGradients() = 0;
    virtual void setTrainingSamples(std::shared_ptr<arma::mat> X, std::shared_ptr<arma::mat> Y){}
    virtual void setTrainingSamples(std::shared_ptr<arma::cube> X, std::shared_ptr<arma::cube> Y){}
    virtual void calGradient() = 0;
    virtual double getLoss() = 0;
    virtual void save(std::string filename) = 0;
    virtual void load(std::string filename) = 0;
    virtual std::shared_ptr<arma::mat> netOutput() = 0;
    virtual ~Net() {
    }
};


