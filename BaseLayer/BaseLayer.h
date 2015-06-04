#pragma once
#include <memory>
#include <armadillo>


class BaseLayer {
public:
    enum ActivationType {softmax, sigmoid, linear, tanh};
    BaseLayer() {}
    BaseLayer(int inputDim0, int outputDim0, ActivationType actType0);
    void save(std::string filename = "BaseLayer");
    void activateUp(std::shared_ptr<arma::mat> input);
    void updatePara(std::shared_ptr<arma::mat> delta_in, double learningRate);
    int inputDim;
    int outputDim;
    int numInstance;
    std::shared_ptr<arma::mat> inputX, inputY, outputY;
    std::shared_ptr<arma::mat> W;
    std::shared_ptr<arma::vec> B;
    std::shared_ptr<arma::mat> delta_out;
    ActivationType actType;
    void initializeWeight();

};

