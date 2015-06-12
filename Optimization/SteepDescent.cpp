#include<iostream>
#include "Optimization.h"
#include <armadillo>

using namespace Optimization;

void SteepDescent::minimize() {
    arma::vec Grad(inputDim);
    int iter = 0;
    if( !quiet ) {
        std::cout << "Gradient Descent Starts !" << std::endl;
        std::cout << "maxIter:" << maxIter << std::endl;
        std::cout << "alpha:" << alpha << std::endl;
    }
    while( iter < maxIter) {


        double value = calValGrad(oldX,grad);
        if( !quiet ) {
            std::cout << "iter:" << iter << "\t" ;
            std::cout << "current gradient norm is:" << arma::norm(Grad) << std::endl;
            std::cout << "current value is:" << arma::value << std::endl; 
        }

//	newX=oldX-alpha*Grad;
        newX = oldX - alpha * Grad;

        if(converged()) break;

        oldX = newX;
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