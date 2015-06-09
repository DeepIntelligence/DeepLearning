#include <memory>
#include <armadillo>
#include "../MatArray/MatArray.h"

struct PoolLayer {
    enum Type { mean, max};
    PoolLayer() {}
    PoolLayer(int poolDim_x, int poolDim_y, Type type0);
    void setInputDim(int, int, int);
    void activateUp(std::shared_ptr<arma::cube> input0);
    void upSampling(std::shared_ptr<arma::cube> detla_in);
    std::shared_ptr<arma::cube> input;
    std::shared_ptr<arma::cube> output;
    std::shared_ptr<arma::Cube<int>> maxIdx_x, maxIdx_y;
    std::shared_ptr<arma::cube> detla_in;
    std::shared_ptr<arma::cube> delta_out;
    Type type;
    int poolDim_x, poolDim_y;
    int inputDim_x;
    int inputDim_y;
    int inputDim_z;
    int outputDim_x, outputDim_y, outputDim_z;
    int inputSize, outputSize;
};