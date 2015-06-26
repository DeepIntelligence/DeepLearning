#pragma once

#include <memory>
#include <vector>
#include <armadillo>
#include <deque>
#include "Globals.h"

namespace Optimization{

struct ObjectFunc{
    ObjectFunc(int dim0 = 0):dim(dim0){}
    int dim;
    std::shared_ptr<arma::vec> x_init;
    virtual double operator()(arma::vec &x, arma::vec &grad) = 0;
};

class LBFGS{
//	typedef double (* evaluateFunc)(const arma::vec x, arma::vec grad, const int n);
public:
    enum LineSearch {Wolfe, Armijo, MoreThuente};
	struct LBFGS_param{ 
            int maxIter; 
            int memoryLimit;
            int maxLineSearch;
            double maxStepSize;
            double minStepSize;
            int saveFrequency;
            std::string saveFileName;
            LBFGS_param(int, int, int, std::string);};
	struct PointValueDeriv {
            double step, value, deriv;
            PointValueDeriv(double step0 = NaN, double value0 = NaN, double deriv0 = NaN) : 
			    step(step0), value(value0), deriv(deriv0) { }
	};
	LBFGS(ObjectFunc &func, LBFGS_param param0, LineSearch method);
	void calDirection();
	void updateParam();
	void calStepLength_Armijo();
        void calStepLength_Wolfe();
        void calStepLength_MoreThuente();
	bool converge();
	void minimize();
        void saveWeight(std::string str);
        double cubicInterp(const LBFGS::PointValueDeriv& p0, const LBFGS::PointValueDeriv& p1);
        ObjectFunc &calValGrad;
        LBFGS_param param;
        double maxIter;
        double step;
        double currValue;
        int memoryLimit;
        LineSearch lineSearchMethod;
// 	s_{k-1} = x_k - x_{k-1}
//  y_{k-1} = (grad_k - grad_{k-1})
	std::deque<arma::vec> s_list, y_list;
// rho_k =1.0 /(y_k^T * s_k)	
	std::deque<double> rho_list;
	std::vector<double> alpha_list;	
        arma::vec direction;
	arma::vec grad, x, x_init, x_new, grad_new;
};

class SteepDescent{
public:
    struct SteepDescent_param{
        SteepDescent_param(double eps0, double step0, int maxIter0):
                            eps(eps0), step(step0), maxIter(maxIter0){}
        double eps;
        double step;
        int maxIter;};
    SteepDescent(ObjectFunc &func, SteepDescent_param param0);
    void minimize();
private:
 //   bool converged();
    double eps;
    double step;
    int maxIter;
    arma::vec grad, grad_new, x, x_new;
    double currValue;
    SteepDescent_param param;
    ObjectFunc &calValGrad;

};




}