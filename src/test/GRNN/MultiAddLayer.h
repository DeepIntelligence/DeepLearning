#pragma once
#include "common.h"
namespace NeuralNet {

    struct MultiAddLayer : public Layer_binaryOp {
        MultiAddLayer(){}
        MultiAddLayer(int in1, int in2, int out, ActivationType actType0):Layer_binaryOp(in1,in2,out),actType(actType0);
        virtual ~MultiAddLayer(){}
        virtual void activateUp();
        virtual void save(std::string filename = "MultiAddLayer");
        virtual void load(std::string filename = "MultiAddLayer");
        virtual void calGrad(std::shared_ptr<arma::mat> delta_in);
	virtual void calGrad(std::shared_ptr<arma::mat> delta_in, int t);	
        ActivationType actType;
        std::shared_ptr<arma::mat> W_one, W_two, B;
        std::shared_ptr<arma::mat> grad_W_one, grad_W_two, grad_B;
        std::shared_ptr<Initializer> initializer_W_one, initializer_W_two, initializer_B; 


    };
}
