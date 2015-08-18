#pragma once
#include <memory>
#include <armadillo>

namespace NeuralNet{
	enum ActivationType {softmax, sigmoid, linear, tanh, ReLU};
	
	inline void ApplyActivation(std::shared_ptr<arma::mat> output, ActivationType actType){
    std::shared_ptr<arma::mat> &p=output;
    arma::mat maxVal = arma::max(*p,0);
    arma::mat sumVal;
    switch(actType) {
    case softmax:
        for (int i = 0; i < p->n_cols; i++) {
                p->col(i)-= maxVal(i);            
        }          
        (*p).transform([](double val) {
            return exp(val);
        });

         sumVal = arma::sum(*p,0);
        for (int i = 0; i < p->n_cols; i++) {            
               p->col(i) /=sumVal(i);
        }
        break;
    case sigmoid:
        (*p).transform([](double val) {
            return 1.0/(1.0+exp(-val));
        });
        break;
    case linear:
        break;
    case ReLU:
        p->transform([](double val) {return val > 0 ? val: 0 ;});
    	break;
    }
}
	inline void GetActivationGradient(std::shared_ptr<arma::mat> in, std::shared_ptr<arma::mat> out, ActivationType actType){
	
    if (actType == softmax) {
        out->ones(in->n_rows,in->n_cols);   
    } else if (actType == sigmoid ) {
        *out = (1 - (*in)) % (*in);        
    } else if ( actType == tanh) {
        *out = (1 - (*in) % (*in));
    } else if ( actType == linear) {
        out->ones(in->n_rows,in->n_cols);
    } else if(actType == ReLU){
        *out = *in;
        out->transform([](double val) {return val > 0 ? 1.0: 0 ;});
    }	
}

}
