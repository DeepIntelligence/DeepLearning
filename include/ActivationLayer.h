#pragma once
#include "common.h"

namespace NeuralNet {

    struct ActivationLayer: public Layer_unitaryOp {

        ActivationLayer(ActivationType actType0) {
        	actType = actType0;
        };
        ActivationType actType;
		virtual void activateUp();
        virtual void calGrad(std::shared_ptr<arma::mat> delta_in);
    };

}
