#pragma once
#include "common.h"
#include "MultiAddLayer.h"
namespace NeuralNet {

    struct RecurrLayer : public MultiAddLayer {
        RecurrLayer(){}
        RecurrLayer(int in1, int in2, int out, ActivationType actType0,
        std::shared_ptr<Initializer> init_W_one, std::shared_ptr<Initializer> init_W_two,
        std::shared_ptr<Initializer> init_B);
        virtual ~RecurrLayer(){}
        void savePrevOutput();
        void savePrevDeltaOutOne();
        std::shared_ptr<arma::mat> getPrevOutput(){ return output_prev;}
        std::shared_ptr<arma::mat> getPrevDeltaOutOne() { return delta_outOne_prev;}
        std::shared_ptr<arma::mat> output_prev;
        std::shared_ptr<arma::mat> delta_outOne_prev;
    };
}
