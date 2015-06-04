#include <memory>
#include <armadillo>
#include "../MatArray/MatArray.h"


class ConvolveLayer {

public:
    enum ActivationType {ReLU, tanh, sigmoid};
    ConvolveLayer(int numFilters0, int filterDim0_x, int filterDim0_y, int stride0);
    void activateUp(std::shared_ptr<arma::cube>);
// upate the parameters and propgate the error down for the lower layer
    void updatePara(std::shared_ptr<arma::cube> delta_upper);
    void initializeWeight();
    void setInputDim(int, int, int);
    void propError(std::shared_ptr<arma::cube> delta_upper);
    int numFilters;
//  every filter is a 4D cube
    MatArray<double>::Mat2DArray_ptr filters;
    std::shared_ptr<arma::cube> delta_out, input, output;
    std::shared_ptr<arma::cube> B;
    int filterDim_x, filterDim_y;
    int inputDim_x;
    int inputDim_y;
    int inputDim_z;
    int outputDim_x, outputDim_y, outputDim_z;
    int stride;
};