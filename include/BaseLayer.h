#pragma once
#include "common.h"
namespace NeuralNet{

struct BaseLayer {
    enum ActivationType {softmax, sigmoid, linear, tanh, ReLU};
    BaseLayer() {}
    BaseLayer(int inputDim0, int outputDim0, ActivationType actType0, std::shared_ptr<Initializer> init_w = nullptr, 
	std::shared_ptr<Initializer> init_B = nullptr, bool dropout = false, double dropr=0.3);
/*  save weights of the layers
 */
    void save(std::string filename = "BaseLayer");
    void load(std::string filename = "BaseLayer");
/*  given the input matrix, perform 
 outputY = sigma (W*input + B), sigma is the activation function
*/    
    void activateUp();
    void activateUp(std::shared_ptr<arma::mat> input);
    
    
/* activate up using given W and B
 */    
    void activateUp(std::shared_ptr<arma::mat> W, std::shared_ptr<arma::vec> B, std::shared_ptr<arma::mat> input);
/*
 given the error propogated from upper layers, update the W and B using gradient descent
 */    
    void updatePara(std::shared_ptr<arma::mat> delta_in, double learningRate);
/*
 calculate the gradient and propogate the error but not update W and B
 */    
    virtual void calGrad(std::shared_ptr<arma::mat> delta_in);
    
    void accumulateGrad(std::shared_ptr<arma::mat> delta_in);
    
    void updatePara_accu(double learningRate);
    
/* randomly initialize weight and bias*/    
    void initializeWeight();    
    int inputDim;
    int outputDim;
    int W_size, B_size, totalSize;
    std::shared_ptr<arma::mat> input, output;
/*  weight and bias for this layer*/
    std::shared_ptr<arma::mat> W, B;
    std::shared_ptr<arma::mat> grad_W, grad_W_accu, grad_B, grad_B_accu;
/* the error propogated from lower layers*/    
    std::shared_ptr<arma::mat> delta_out;
    bool dropOutFlag;
    double dropOutRate;
    std::shared_ptr<Initializer> initializer_W, initializer_B; 
	arma::mat dropOutMat;
    ActivationType actType;
    Random_Bernoulli<double> *randomGen;
    void vectoriseGrad(std::shared_ptr<arma::vec> V);
    void deVectoriseWeight(std::shared_ptr<arma::vec> V);
    void vectoriseWeight(std::shared_ptr<arma::vec> V);
    void vectoriseGrad(double *ptr, size_t offset);
    void deVectoriseWeight(double *ptr, size_t offset);
    void vectoriseWeight(double *ptr, size_t offset);
    void applyActivation();
	
    void fill_Bernoulli(double *, int size); 
};

}
