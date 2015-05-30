#include <memory>
#include <armadillo>


class ConvolveLayer{

 public:
  ConvolveLayer(){}
  activateUp(std::shared_ptr<arma::cube> input); 	
 private:
  int numFilters;	
  std::shared<arma::mat> W;
  std::shared<arma::vec> B;
  int filterDim;
  int imageDim;
  int stride;	
	
	
};