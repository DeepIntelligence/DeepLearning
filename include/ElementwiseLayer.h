#pragma once
#include "common.h"

namespace NeuralNet {

    struct ElementwiseLayer: public Layer_binaryOp {

        ElementwiseLayer() {
            //we only need to assign memory to the output
            delta_outOne = std::make_shared<arma::mat>();
            delta_outTwo = std::make_shared<arma::mat>();
            output = std::make_shared<arma::mat>();
        };
        virtual void activateUp();
        virtual void calGrad(std::shared_ptr<arma::mat> delta_in, int timePoint);
		virtual void calGrad(std::shared_ptr<arma::mat> delta_in);
    };

}
