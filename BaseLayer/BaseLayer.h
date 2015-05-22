#pragma once
#include <memory>
#include <armadillo>


class BaseLayer{
 public:
  enum ActivationType{softmax, sigmoid, linear};  
  BaseLayer(){}
  BaseLayer(int inputDim0, int outputDim0, ActivationType actType0);
  void save(std::string filename = "BaseLayer");
  void activateUp();
  int inputDim;
  int outputDim;
  int numInstance;
  std::shared_ptr<arma::mat> inputX, inputY, outputY;
  std::shared_ptr<arma::mat> W;
  std::shared_ptr<arma::vec> B;
  ActivationType actType;  
  void initializeWeight();
    
};

