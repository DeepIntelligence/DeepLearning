#pragma once
#include "common.h"
namespace NeuralNet {

    struct ElementMultiAddLayer : public Layer_binaryOp {
        ElementMultiAddLayer();
        virtual ~ElementMultiAddLayer(){}
        virtual void activateUp();
        virtual void calGrad(std::shared_ptr<arma::mat> delta_in);
		virtual void calGrad(std::shared_ptr<arma::mat> delta_in, int t);
		void saveWeightMem();
        std::shared_ptr<arma::mat> W_one, W_two;
        std::shared_ptr<arma::mat> grad_W_one, grad_W_two;
		std::vector<std::shared_ptr<arma::mat>> W_one_mem, W_two_mem;

    };
}
