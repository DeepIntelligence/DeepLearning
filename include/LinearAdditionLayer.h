#pragma once
#include "common.h"

namespace NeuralNet {

    struct LinearAdditionLayer: public Layer_binaryOp {

        LinearAdditionLayer() {
            output = std::make_shared<arma::mat>();
        }
        virtual void activateUp();
        virtual void calGrad(std::shared_ptr<arma::mat> delta_in);
		

    };

}
