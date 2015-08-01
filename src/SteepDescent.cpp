#include<iostream>
#include "optimization.h"
#include <armadillo>

using namespace Optimization;

SteepDescent::SteepDescent(ObjectFunc &func, SteepDescent_param param0):
                            calValGrad(func), param(param0){

    maxIter = param.maxIter;
    eps = param.eps;
    step = param.step;
    
    x.randn(calValGrad.dim);
    
    currValue = calValGrad(x, grad);
}


void SteepDescent::minimize() {
//    arma::vec Grad(inputDim);
    int iter = 0;
//    if( !quiet ) {
        std::cout << "Gradient Descent Starts !" << std::endl;
        std::cout << "maxIter:" << maxIter << std::endl;
//        std::cout << "alpha:" << alpha << std::endl;
//    }
    while( iter < maxIter) {

        x_new = x - step * grad;
        double currValue = calValGrad(x_new,grad_new);
//        if( !quiet ) {
            std::cout << "iter:" << iter << "\t" ;
            double gradNorm =  arma::norm(grad_new);
            std::cout << "current gradient norm is:" << gradNorm << std::endl;
            std::cout << "current value is:" << currValue << std::endl; 
//        }
        x = x_new;
        grad = grad_new;

        if ( gradNorm < eps) break;
        iter++;

    }


}

/*
bool GradDescent::converged() {
    arma::vec diff;
    diff = newX - oldX;
    return arma::norm(diff) < eps;
}
 */

//GradDescent::~GradDescent(){}