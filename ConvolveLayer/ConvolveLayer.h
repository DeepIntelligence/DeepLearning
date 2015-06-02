#include <memory>
#include <armadillo>
#include "../MatArray/MatArray.h"


class ConvolveLayer {

public:
    enum ActivationType {ReLU, tanh, sigmoid};
    ConvolveLayer(int numFilters) {}
    void activateUp();
// upate the parameters and propgate the error down for the lower layer
    void updatePara(std::shared<arma::cube> delta_upper);
    void initializeWeight();
    void setInput(std::shared<arma::cube> input0);
    void propError(std::shared<arma::cube> delta_upper);
private:
    int numFilters;
//  every filter is a 4D cube
    MatArray<double>::Mat2DArray_ptr filters;
    std::shared<arma::cube> delta_out, input, output;
    std::shared_ptr<arma::cube> B;
    int filterDim_x, filterDim_y;
    int inputDim_x;
    int inputDim_y;
    int inputDim_z;
    int outputDim_x, outputDim_y, outputDim_z;
    int stride;
    int numFilters;
};