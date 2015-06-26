#pragma once
#include <armadillo>
#include "../BaseLayer/BaseLayer.h"
#include "../Optimization/optimization.h"

struct TrainingPara {

    TrainingPara(double eps0=1e-6, int NEpoch0 = 500,
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
    MultiLayerPerceptron(int numLayers0, std::vector<int> dimensions0, std::shared_ptr<arma::mat> trainingX0,
                         std::shared_ptr<arma::mat> trainingY0, TrainingPara trainingPara);

    enum ParaType{gradient, weight};
    void train();
    void initialize();
/* forward pass*/    
    void feedForward(std::shared_ptr<arma::mat>);
/* back propogate the error to update the parameters*/    
    void backProp(std::shared_ptr<arma::mat>);
    void test(std::shared_ptr<arma::mat> trainingX,std::shared_ptr<arma::mat> trainingY);
/* calculate the numerical gradient for testing*/    
    void calNumericGrad(std::shared_ptr<arma::mat> trainingX,std::shared_ptr<arma::mat> trainingY);
    void calGrad(std::shared_ptr<arma::mat> trainingX);
    void vectoriseGrad(arma::vec &grad);
    void deVectoriseWeight(arma::vec &x);
    void vectoriseWeight(arma::vec &x);
private:
    bool converge();
    TrainingPara trainingPara;
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
    std::shared_ptr<arma::mat> outputY;
    int totalDim;

};

class MLPTrainer:public Optimization::ObjectFunc{
public:
    MLPTrainer(MultiLayerPerceptron &MLP);
    virtual double operator()(arma::vec &x, arma::vec &grad);
//    std::shared_ptr<arma::vec> x_init;
private:
    MultiLayerPerceptron  &MLP;
};