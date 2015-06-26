#include <memory>
#include <armadillo>
#include "../MatArray/MatArray.h"


struct ConvolveLayer {
    enum ActivationType {ReLU, tanh, sigmoid, linear};
    ConvolveLayer(int numFilters0, int filterDim0_x, int filterDim0_y, int stride0, ActivationType type0);
    void activateUp(std::shared_ptr<arma::cube>);
// upate the parameters and propgate the error down for the lower layer
    void updatePara(std::shared_ptr<arma::cube> delta_upper, double learningRate);
    void calGrad(std::shared_ptr<arma::cube> delta_upper);
    void calGrad_matrixMethod(std::shared_ptr<arma::cube> delta_upper);
    void initializeWeight();
    void setInputDim(int, int, int);
    void propError(std::shared_ptr<arma::cube> delta_upper);
    void vectoriseGrad(double *ptr, size_t offset);
    void deVectoriseWeight(double *ptr, size_t offset);
    void vectoriseWeight(double *ptr, size_t offset);
    void convolve_naive(std::shared_ptr<arma::cube> input);
    void im2col(std::shared_ptr<arma::cube> input, std::shared_ptr<arma::mat> &output);
    void col2im(std::shared_ptr<arma::mat> input, std::shared_ptr<arma::cube> &output);
    void convolve_matrixMethod(std::shared_ptr<arma::cube> input);
    
    int numFilters;
//  every filter is a 4D cube
//    MatArray<double>::Mat2DArray_ptr filters, grad_W;
    Tensor_4D::ptr filters, grad_W;
    std::shared_ptr<arma::cube> delta_out, input, output;
    std::shared_ptr<arma::cube> B, grad_B;
    std::shared_ptr<arma::mat> filters2D, input2D, grad_W2D;
    
    int filterDim_x, filterDim_y;
    int inputDim_x;
    int inputDim_y;
    int inputDim_z;
    int inputSize;
    int outputSize;
    int outputDim_x, outputDim_y, outputDim_z;
    int stride;
    int W_size, B_size, totalSize;
    ActivationType type;
};