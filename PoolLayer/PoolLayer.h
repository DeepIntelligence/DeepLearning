#include <memory>
#include <armadillo>
#include "../MatArray/MatArray.h"

class PoolLayer{
	
 public:
  enum Type { mean, max};
  PoolLayer(){}
  PoolLayer(int poolDim0, Type type0, MatArray<double>::Mat1DArray_ptr inputX0):
            poolDim(poolDim0),type(type0), inputX(inputX0){}
  void activateUp();
  MatArray<double>::Mat1DArray_ptr inputX;
  MatArray<double>::Mat1DArray_ptr outputX;
  MatArray<int>::Mat1DArray_ptr maxIdx_x, maxIdx_y;
  Type type;
  int poolDim;	
	
};