#include <memory>
#include <armadillo>
#include "../MatArray/MatArray.h"


class ConvolveLayer{

 public:
  enum ActivationType{ReLU, tanh, sigmoid};
  ConvolveLayer(int numFilters, ){}
  activateUp(); 	
  updatePara(MatArray<double>::Mat1DArray delta_upper);
  
 private:
  int numFilters;	
//  every filter is a 4D cube   
  MatArray<double>::Mat2DArray filters;
  MatArray<double>::Mat1DArray input;
  arma::mat B;
  int filterDim;
  int imageDim;
  int stride;	
  int numFilters;
	
	
};