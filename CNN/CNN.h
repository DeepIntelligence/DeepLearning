#include <memory>
#include <armadillo>
#include "../BaseLayer/BaseLayer.h"
#include "../ConvolveLayer/ConvolveLayer.h"
#include "../PoolLayer/PoolLayer.h"
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


class CNN {
public:
    CNN(){};
    CNN(std::shared_ptr<arma::cube> trainingX0, std::shared_ptr<arma::mat> trainingY0, int nChanel0);
    void train();
    void setTrainingData(std::shared_ptr<arma::cube> trainingX0, std::shared_ptr<arma::mat> trainingY0, int nChanel0);
    void feedForward(std::shared_ptr<arma::cube>);
    void backProp(std::shared_ptr<arma::mat>);

    std::vector<PoolLayer> poolLayers;
    std::vector<ConvolveLayer> convoLayers;
    std::vector<BaseLayer> FCLayers;
    int numInstance;
    std::shared_ptr<arma::cube> trainingX;
    std::shared_ptr<arma::mat> trainingY, output;
    int nChanel;
    TrainingPara trainingPara;
    int inputDim_x, inputDim_y;
    int numFCLayers;
    int numCLayers;
};