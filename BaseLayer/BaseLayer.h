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
  arma::mat W;
  arma::vec B;
  std::shared_ptr<arma::mat> inputX;
  std::shared_ptr<arma::mat> inputY;
  std::shared_ptr<arma::mat> outputY;
  ActivationType actType;  
  void initializeWeight();
    
};

