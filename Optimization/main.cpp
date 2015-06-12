#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <armadillo>
#include "optimization.h"
#include <vector>

void loadData_MNIST(std::shared_ptr<arma::mat> X,
                    std::shared_ptr<arma::mat> Y);

using namespace Optimization;

struct MyFunc: public ObjectFunc{

    MyFunc(int dim):ObjectFunc(dim){}
    
    virtual double operator()(arma::vec &x, arma::vec &grad){
        
        grad = 2.0* x;
        return arma::as_scalar(x.st() * x);
    }
    
};

struct MyFunc2: public ObjectFunc{

    arma::mat W;
    arma::vec b;
    MyFunc2(int dim):ObjectFunc(dim){
        W.eye(dim,dim);
        b = arma::randu(dim);
    }
    
    virtual double operator()(arma::vec &x, arma::vec &grad){
        
        
        grad = W * x + b;
        return arma::as_scalar(0.5 * x.st() * W * x + b.st()*x);
    }
    
};

struct MyFunc3: public ObjectFunc{

    MyFunc3(int dim = 2):ObjectFunc(dim){
        
    }
    
    virtual double operator()(arma::vec &x, arma::vec &grad){
        
        double F = 10.0 * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0]) + (x[0] - 1) * (x[0] - 1);
        
        grad.resize(2);
        grad[0] = -40.0 * x[0] * x[1] + 40.0 * pow(x[0],3.0) + 2 * x[0] - 2;
        grad[1] = 20 * x[1] - 20.0 * x[0] * x[0];
        return F;
    }
    
};

int main(int argc, char *argv[]) {

    int dim = 5;
    
    MyFunc myFunc(dim);
    
    arma::vec x(dim);
    arma::vec grad;
//    x.randu();
    
//    x.print();
//    std::cout << myFunc(x,grad) << std::endl;
//    grad.print();
    LBFGS::LBFGS_param param;
    param.maxIter = 10;
    param.memoryLimit = 20;
//    std::cout << "test case 1" << std::endl;
//    LBFGS lbfgs_opt(myFunc,param);
//    lbfgs_opt.minimize();
    
    std::cout << "test case 2" << std::endl;
    MyFunc2 myFunc2(dim);
    LBFGS lbfgs_opt2(myFunc2,param,LBFGS::Armijo);
    lbfgs_opt2.minimize();
    
    std::cout << "test case 3" << std::endl;
    MyFunc3 myFunc3;
    LBFGS lbfgs_opt3(myFunc3,param,LBFGS::Armijo);
    lbfgs_opt3.minimize();
    
    
    return 0;
}


void loadData_MNIST(std::shared_ptr<arma::mat> X,
                    std::shared_ptr<arma::mat> Y) {

    std::string filename_base("../MNIST/data");
    std::string filename;
    char tag[50];
    char x;
    int count;
    int numFiles = 10;
    int featSize = 28*28;
    int labelSize = 10;
    int numSamples = 1000;
    X->set_size(numFiles*numSamples,featSize);
    Y->set_size(numFiles*numSamples,labelSize);
    Y->fill(0);
//  std::cout << Y.Len() << std::endl;
//  std::cout << X.NumR() << std::endl;
//  std::cout << X.NumC() << std::endl;

    for (int i = 0 ; i < numFiles ; i++) {
        sprintf(tag,"%d",i);
        filename=filename_base+(std::string)tag;
        std::cout << filename << std::endl;
        std::ifstream infile;
        infile.open(filename,std::ios::binary | std::ios::in);
        if (infile.is_open()) {

            for (int j = 0 ; j < numSamples ; j++) {

                for (int k =0 ; k <featSize; k ++) {
                    infile.read(&x,1);
//        std::cout << x << std::endl;
                    (*X)(j+i*numSamples,k)=(unsigned char)x;
                }
                (*Y)(j+i*numSamples,i) = 1;
            }

        } else {
            std::cout << "open file failure!" << std::endl;
        }

// for (int j = 0 ; j < numSamples ; j++){
//       for (int k =0 ; k <featSize; k ++){

//	           std::cout << x << std::endl;
//	   std::cout<<  (*X)(j,k) << " ";
//	   }
//	   }

        std::cout << "dataloading finish!" <<std::endl;
    }
}
