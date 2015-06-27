#include <memory>
#include <armadillo>
#include "../BaseLayer/BaseLayer.h"
#include "../ConvolveLayer/ConvolveLayer.h"
#include "../PoolLayer/PoolLayer.h"
#include "../Optimization/optimization.h"

namespace NeuralNet{

struct TrainingPara {

    TrainingPara(double eps0=1e-6, int NEpoch0 = 500,
                 int miniBatchSize0 = 10, double alpha0 = 0.1, int save = 50, bool load = false):
        eps(eps0),NEpoch(NEpoch0),
        miniBatchSize(miniBatchSize0), alpha(alpha0), saveFrequency(save), loadFlag(load) {}
    double eps;
    int NEpoch;
    int miniBatchSize;
    double alpha;
    int saveFrequency;
    bool loadFlag;
//  Method method;
    void print() const {

        std::cout << eps << "\t";
        std::cout << NEpoch << "\t";
        std::cout << miniBatchSize << "\t";
        std::cout << alpha << std::endl;

    }
};


class CNN {
    friend class CNNTrainer;
public:
    CNN(){};
    CNN(std::shared_ptr<arma::cube> trainingX0, std::shared_ptr<arma::mat> trainingY0, int nChanel0, TrainingPara trainingPara0);
    void train();
    void setTrainingData(std::shared_ptr<arma::cube> trainingX0, std::shared_ptr<arma::mat> trainingY0, int nChanel0);
    void feedForward(std::shared_ptr<arma::cube>);
    void backProp(std::shared_ptr<arma::mat>);
    void test(std::shared_ptr<arma::cube> testX0, std::shared_ptr<arma::mat> testY0);
    double calLayerError(std::shared_ptr<arma::cube> delta);
    void calNumericGrad(std::shared_ptr<arma::cube>, std::shared_ptr<arma::mat>);
    void calGrad(std::shared_ptr<arma::mat> trainingX);    
    void vectoriseGrad(arma::vec &grad);
    void deVectoriseWeight(arma::vec &x);
    void vectoriseWeight(arma::vec &x);
    void saveWeight(std::string str = "cnn_weights.dat");
    void loadWeight(std::string str = "cnn_weights.dat");
    
    bool testGrad;
    std::vector<PoolLayer> poolLayers;
    std::vector<ConvolveLayer> convoLayers;
    std::vector<BaseLayer> FCLayers;
    int numInstance;
    std::shared_ptr<arma::cube> trainingX;
    std::shared_ptr<arma::mat> trainingY, output;
    int nChanel;
    TrainingPara trainingPara;
    int inputDim_x, inputDim_y;
    int outputDim;
    int numFCLayers;
    int numCLayers;
    int totalDim;
};

class CNNTrainer:public Optimization::ObjectFunc{
public:
    CNNTrainer(CNN &CNN);
    void gradientChecking();
    virtual double operator()(arma::vec &x, arma::vec &grad);
//  std::shared_ptr<arma::vec> x_init;
private:
    CNN  &cnn;
};

}