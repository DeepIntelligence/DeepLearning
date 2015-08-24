#pragma once
#include <memory>
#include <armadillo>
#include <iostream>
#include <vector>
#include "RNN.h"
#include "common.h"
namespace NeuralNet {

    class ElmanRL: public RNN {
        
    public:
        ElmanRL(DeepLearning::NeuralNetParameter);
		virtual ~ElmanRL(){}
        
        // implementing methods required by Net interface
        virtual arma::mat forwardInTime(std::shared_ptr<arma::mat> x);        
        void backward();
    };
}


