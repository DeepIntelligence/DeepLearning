#include <memory>
#include <armadillo>
#include "../MatArray/MatArray.h"

class PoolLayer {

public:
    enum Type { mean, max};
    PoolLayer() {}
    PoolLayer(int poolDim_x, int poolDim_y, Type type0);
    void setInput(std::shared<arma::cube> input0);
    void activateUp();
    void upSampling(std::shared<arma::cube> detla_in);
    std::shared<arma::cube> input;
    std::shared<arma::cube> output;
    std::shared<arma::Cube<int>> maxIdx_x, maxIdx_y;
    std::shared<arma::cube> detla_in;
    std::shared<arma::cube> delta_out;
    Type type;
    int poolDim_x, poolDim_y;
    int inputDim_x;
    int inputDim_y;
    int inputDim_z;
    int outputDim_x, outputDim_y, outputDim_z;
};