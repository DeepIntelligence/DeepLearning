#pragma once
#include <armadillo>
#include "BaseLayer.h"
#include "optimization.h"
#include "DeepLearning.pb.h"
namespace NeuralNet{

struct TrainingPara_MLP {

    TrainingPara_MLP(double eps0=1e-6, int NEpoch0 = 500,
                 int miniBatchSize0 = 10, double alpha0 = 0.1):
        eps(eps0),NEpoch(NEpoch0),
        miniBatchSize(miniBatchSize0), alpha(alpha0) {}


    double eps;
    int NEpoch;
    int miniBatchSize;
    double alpha;
//  Method method;
    void print() const {

        std::cout << eps << "\t";
        std::cout << NEpoch << "\t";
        std::cout << miniBatchSize << "\t";
        std::cout << alpha << std::endl;

    }
};



class MultiLayerPerceptron {
    friend class MLPTrainer;
public:
    MultiLayerPerceptron(int numLayers0, std::vector<int> dimensions0, TrainingPara_MLP trainingPara);
    MultiLayerPerceptron(DeepLearning::NeuralNetParameter);
    MultiLayerPerceptron(int numLayers0, std::vector<int> dimensions0, std::shared_ptr<arma::mat> trainingX0,
                         std::shared_ptr<arma::mat> trainingY0, TrainingPara_MLP trainingPara);
    ~MultiLayerPerceptron(){}
    enum ParaType{gradient, weight};
    void train();
    void initialize();
/* forward pass*/    
    void feedForward(std::shared_ptr<arma::mat>);
/* back propogate the error to update the parameters*/    
    void backProp(std::shared_ptr<arma::mat>, double learningRate);
    void test(std::shared_ptr<arma::mat> trainingX,std::shared_ptr<arma::mat> trainingY);
/* calculate the numerical gradient for testing*/    
    void calNumericGrad(std::shared_ptr<arma::mat> trainingX,std::shared_ptr<arma::mat> trainingY);
    void calGrad(std::shared_ptr<arma::mat> trainingX);
    void vectoriseGrad(arma::vec &grad);
    void deVectoriseWeight(arma::vec &x);
    void vectoriseWeight(arma::vec &x);
    void save(std::string filename);
    void setTrainingSample(std::shared_ptr<arma::mat> X, std::shared_ptr<arma::mat> Y);
    std::shared_ptr<arma::mat> getNetOutput(){return netOutput;}
private:
    bool converge();
    TrainingPara_MLP trainingPara;
    DeepLearning::NeuralNetParameter neuralNetPara;
	int numLayers;
    int numInstance;
    bool testGrad;
/**the collection of Base layers*/
    std::vector<BaseLayer> layers;
/**training data, input and label*/    
    std::shared_ptr<arma::mat> trainingX;
    std::shared_ptr<arma::mat> trainingY;
/* dimension parameters for each layer*/    
    std::vector<int> dimensions;
/* network output*/    
    std::shared_ptr<arma::mat> netOutput;
    int totalDim;

};

class MLPTrainer:public Optimization::ObjectFunc{
public:
    MLPTrainer(MultiLayerPerceptron &MLP);
    ~MLPTrainer(){}
    virtual double operator()(arma::vec &x, arma::vec &grad);
//    std::shared_ptr<arma::vec> x_init;
private:
    MultiLayerPerceptron  &MLP;
};
}
