#pragma once
#include <armadillo>
#include "../BaseLayer/BaseLayer.h"


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
public:
    MultiLayerPerceptron(int numLayers0, std::vector<int> dimensions0, std::shared_ptr<arma::mat> trainingX0,
                         std::shared_ptr<arma::mat> trainingY0, TrainingPara trainingPara);

    void train();
    void initialize();
    void feedForward(std::shared_ptr<arma::mat>);
    void backProp(std::shared_ptr<arma::mat>);
    void test(std::shared_ptr<arma::mat> trainingX,std::shared_ptr<arma::mat> trainingY);
private:
    bool converge();
    TrainingPara trainingPara;
    int numLayers;
    int numInstance;
    std::vector<BaseLayer> layers;
    std::shared_ptr<arma::mat> trainingX;
    std::shared_ptr<arma::mat> trainingY;
    std::vector<int> dimensions;
    std::shared_ptr<arma::mat> outputY;


};