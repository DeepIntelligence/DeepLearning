#include "optimization.h"

using namespace Optimization;

void LBFGS::minimize(){
	
    int iter = 0;
    double grad_norm = arma::norm(grad);
    std::cout << "initial gradient norm is:" << grad_norm << std::endl;
    std::cout << "current value is:" << currValue << std::endl;
    while (true) {
            
        std::cout << "LBFGS iteration:" << iter << std::endl;

        calDirection();
        if (lineSearchMethod == Armijo){
            calStepLength_armijo();
        } else if (lineSearchMethod == Wolf){
            calStepLength_wolf();
        }
	
	updateParam();
        double grad_norm = arma::norm(grad);
        std::cout << "current gradient norm is:" << grad_norm << std::endl;
        std::cout << "current value is:" << currValue << std::endl;
        std::cout << "step length is:" << step << std::endl;
        x.print("current x is:");
        if (grad_norm < 1e-3) break;
//	if (converge()) break;
        iter++;
		
    }
	
}

LBFGS::LBFGS(ObjectFunc& func, LBFGS_param param0, LineSearch method):
            calValGrad(func), param(param0), lineSearchMethod(method){
    
    maxIter = param.maxIter;
    memoryLimit = param.memoryLimit;
    
    x.randu(calValGrad.dim);
    
    currValue = calValGrad(x,grad);
    
    alpha_list.reserve(memoryLimit);
//    rho_list.reserve(memoryLimit);

}


void LBFGS::calDirection(){
	
    
    direction = -1.0 * grad;
    int count = (int) s_list.size();
	
    if (count != 0) {
	for (int i = count -1; i >= 0; i--) {
            alpha_list[i] = rho_list[i] * arma::as_scalar( s_list[i].st() * direction );			
            direction -= alpha_list[i] * y_list[i];
	}

// here the scalar is from nonlinear optimization equation (7.20)		
        double scalar = rho_list[count - 1] / arma::as_scalar( y_list[count - 1].st() * y_list[count - 1] );
	direction *= scalar;
		
	for (int i = 0; i < count; i++) {
            double beta = rho_list[i] * arma::as_scalar(y_list[i].st() * direction);
            direction += s_list[i] * (alpha_list[i] - beta); 
	}
		
	}
    direction.print("direction is:");
}

void LBFGS::updateParam(){
	
	int listSize = (int) s_list.size();
	arma::vec s_next, y_next;
	
	if (listSize == memoryLimit){
// if memory limit is reached		
            s_next = s_list.front();
            s_list.pop_front();
            y_next = y_list.front();
            y_list.pop_front();
            rho_list.pop_front();
	}
	
	s_next = x_new - x;;
	y_next = grad_new - grad;
	
	double rho = arma::as_scalar( s_next.st() * y_next );
	
	s_list.push_back(s_next);
	y_list.push_back(y_next);
	rho_list.push_back(rho);
	
	x = x_new;
	grad = grad_new;
}


void LBFGS::calStepLength_armijo(){
    
    double dirDeriv = arma::as_scalar(direction.st() * grad);
//    initial step
    step = 1.0;
    double eta = 0.8;
    double tau = 0.5;
    
    x_new = x + step * direction;
    double value = calValGrad(x_new, grad_new);
        
    while (value > currValue + eta * step * dirDeriv){
        step = tau * step;
        x_new = x + step * direction;
        value = calValGrad(x_new, grad_new);   
    }

    currValue = value;
}

void LBFGS::calStepLength_wolf(){
//	dirDeriv is (grad f)^T * Direction
    double value;
    double dirDeriv = arma::as_scalar(direction.st() * grad);
        
    if (dirDeriv >= 0) {		
	abort();
    }

//	the two criteria parameter	
    double c1 = 1e-4;
    double c2 = 0.9;

    PointValueDeriv prev(0, currValue, dirDeriv);
    PointValueDeriv aLo, aHi;
    bool done = false;
	
//	initial step

    double step = 1.0;
    double step_star = step;
	
    int iter = 0;
	
    for (int iter = 0; iter < maxIter ; iter++) {
	x_new = x + step * direction;
        double value = calValGrad(x_new, grad_new);
        double dirDeriv_new = arma::as_scalar(direction.st() * grad_new);
		
	PointValueDeriv curr(step, value, dirDeriv_new);
		
	if ((curr.value > currValue + c1 * step * dirDeriv) || (iter > 0 && curr.value >= prev.value)) {
            aLo = prev;
            aHi = curr;
            break;
	}
		
	if (fabs(curr.deriv) <= -c2 * dirDeriv) {
            step_star = curr.step;
            done = true;
            break;
	}
	if (curr.deriv >= 0) {			
            aLo = curr;
            aHi = prev;
            break;
	}
		
        step *= 2.0;
    }

	double minChange = 0.01;

// this loop is the "zoom" procedure described in Nocedal & Wright

    for (int iter = 0; iter < maxIter; iter++) {
        if (aLo.step == aHi.step) return;
        PointValueDeriv left = aLo.step < aHi.step ? aLo : aHi;
        PointValueDeriv right = aLo.step < aHi.step ? aHi : aLo;

        if (IsInf(left.value) || IsInf(right.value)) {
            step = (aLo.step + aHi.step) / 2;
        } else if (left.deriv > 0 && right.deriv < 0) {
	// interpolating cubic would have max in range, not min (can this happen?)
	// set a to the one with smaller value
            step = aLo.value < aHi.value ? aLo.step : aHi.step;
        } else {
            step = cubicInterp(aLo, aHi);
        }

		// this is to ensure that the new point is within bounds
		// and that the change is reasonably sized
        double ub = (minChange * left.step + (1 - minChange) * right.step);
        if (step > ub) step = ub;
        double lb = (minChange * right.step + (1 - minChange) * left.step);
        if (step < lb) step = lb;
            
        x_new = x + step * direction;
        value = calValGrad(x_new, grad_new);
        if (IsNaN(value)) {
            cerr << "Got NaN." << endl;
            abort();
	}

	double dirDeriv_new = arma::as_scalar(direction.st() * grad_new);
	PointValueDeriv curr(step, value, dirDeriv_new);

	if ((curr.value > currValue + c1 * step *dirDeriv) || (curr.value >= aLo.value)) {
            aHi = curr;
	} else {

         // if found the optimal length   
            if (fabs(curr.deriv) <= -c2 * dirDeriv) {
            step_star = step;
            currValue = value;
            done = true;
            return;
            } 
	
            if (curr.deriv * (aHi.step - aLo.step) >= 0) aHi = aLo;
            
            aLo = curr;
	}
	}
        
        std::cout << "step_star not found! will use " << step_star <<std::endl;
}

/// <summary>
/// Cubic interpolation routine from Nocedal and Wright
/// </summary>
/// <param name="p0">first point, with value and derivative</param>
/// <param name="p1">second point, with value and derivative</param>
/// <returns>local minimum of interpolating cubic polynomial</returns>
double LBFGS::cubicInterp(const LBFGS::PointValueDeriv& p0, const LBFGS::PointValueDeriv& p1) {
	double t1 = p0.deriv + p1.deriv - 3 * (p0.value - p1.value) / (p0.step - p1.step);
	double sign = (p1.step > p0.step) ? 1 : -1;
	double t2 = sign * sqrt(t1 * t1 - p0.deriv * p1.deriv);
	double num = p1.deriv + t2 - t1;
	double denom = p1.deriv - p0.deriv + 2 * t2;
	return p1.step - (p1.step - p0.step) * num / denom;
}

bool LBFGS::converge(){
    return false;
}