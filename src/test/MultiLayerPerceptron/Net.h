#pragma once
#include "common.h"

class Net {
public:
    virtual void applyUpdates(std::vector<std::shared_ptr<arma::mat>>) = 0;
    virtual std::vector<std::shared_ptr<arma::mat>> netGradients() = 0;
    virtual void setTrainingSamples(std::shared_ptr<arma::mat> X, std::shared_ptr<arma::mat> Y) = 0;
    virtual void calGradient() = 0;

    virtual ~Net() {
    }
};


