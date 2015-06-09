#pragma once
#include <memory>
#include <armadillo>


struct BaseLayer {
    enum ActivationType {softmax, sigmoid, linear, tanh};
    BaseLayer() {}
    BaseLayer(int inputDim0, int outputDim0, ActivationType actType0);
/*  save weights of the layers
 */
    void save(std::string filename = "BaseLayer");
/*  given the input matrix, perform 
 outputY = sigma (W*input + B), sigma is the activation function
*/    
    void activateUp(std::shared_ptr<arma::mat> input);
/*
 given the error propogated from upper layers, update the W and B using gradient descent
 */    
    void updatePara(std::shared_ptr<arma::mat> delta_in, double learningRate);
/* randomly initialize weight and bias*/    
    void initializeWeight();    
    int inputDim;
    int outputDim;
    std::shared_ptr<arma::mat> inputX, inputY, outputY;
/*  weight and bias for this layer*/
    std::shared_ptr<arma::mat> W;
    std::shared_ptr<arma::vec> B;
/* the error propogated from lower layers*/    
    std::shared_ptr<arma::mat> delta_out;
    ActivationType actType;
    

};

