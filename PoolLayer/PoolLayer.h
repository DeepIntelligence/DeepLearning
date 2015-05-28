#include <memory>
#include <armadillo>


class PoolLayer{
	
 public:
  enum Type { mean, max};
  PoolLayer(int poolDim0, Type type0):poolDim(poolDim0),type(type0){}
  void activateUp(std::shared_ptr<arma::cube> inputX);
  std::shared_ptr<arma::cube> outputX;
  Type type;
  int poolDim;	
	
};