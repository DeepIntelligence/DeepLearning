#include <cmath>
#include "ConvolveLayer.h"


ConvolveLayer::ConvolveLayer(int numFilters0, int filterDim0_x, int filterDim0_y,
                             int stride0, ActivationType type0) {
    numFilters = numFilters0;
    filterDim_x = filterDim0_x;
    filterDim_y = filterDim0_y;
    stride = stride0;
    type = type0;
}

void ConvolveLayer::setInputDim(int inputDim0_x, int inputDim0_y, int inputDim0_z){

    inputDim_x = inputDim0_x;
    inputDim_y = inputDim0_y;
    inputDim_z = inputDim0_z;
    inputSize = inputDim_x * inputDim_y * inputDim_z;
    outputDim_x = inputDim_x / stride;
    outputDim_y = inputDim_y / stride;
    outputDim_z = numFilters;
    outputSize = outputDim_x * outputDim_y * outputDim_z;

    initializeWeight();
}

void ConvolveLayer::initializeWeight() {
    B = std::make_shared<arma::cube>(outputDim_x,outputDim_y, outputDim_z,arma::fill::randu);
    B->transform([&](double val){return (val-0.5)/sqrt(outputSize);});
    filters = MatArray<double>::build(numFilters, inputDim_z, filterDim_x, filterDim_y);
    for (int i = 0; i < numFilters; i++){
        for (int j = 0; j < inputDim_z; j++){
            (*filters)[i][j].randu();
            (*filters)[i][j].transform([&](double val){return (val-0.5)/sqrt(outputSize);}) ;   
        }
    };
    
}

void ConvolveLayer::activateUp(std::shared_ptr<arma::cube> subInput) {
//  after convolution, the size of the image will usually shrink due to stride
    
    input = subInput;
    int halfSize_x = filterDim_x / 2;
    int halfSize_y = filterDim_y / 2;
    output = std::make_shared<arma::cube>(outputDim_x, outputDim_y, outputDim_z, arma::fill::zeros);
//    output->zeros();
    for (int filterIdx = 0; filterIdx < numFilters; filterIdx++) {
        for (int imIdx_z = 0; imIdx_z < inputDim_z ; imIdx_z++) {
            for (int j = 0; j < inputDim_x; j+=stride ) {
                for (int k = 0; k < inputDim_y; k+=stride) {
                    for (int m = 0; m < filterDim_x; m++) {
                        for (int n = 0; n < filterDim_y; n++) {
                            int imIdx_x = j - ( m - halfSize_x );
                            int imIdx_y = k - ( n - halfSize_y );
                            if (imIdx_x >=0 && imIdx_x < inputDim_x && imIdx_y >=0 && imIdx_y < inputDim_y)
                                (*output)(j/stride,k/stride,filterIdx) += (*input)(imIdx_x, imIdx_y,imIdx_z) * (*filters)[filterIdx][imIdx_z](m,n);
                        }
                    }
                }
            }
        }
    }
    
    for (int filterIdx = 0; filterIdx < numFilters; filterIdx++) {
        for (int j = 0; j < outputDim_x; j++ ) {
            for (int k = 0; k < outputDim_y; k++) {
                (*output)(j, k,filterIdx) += (*B)(j, k,filterIdx);
                
            }
        }
    }

//    output->save("convoLayer_output_before.dat", arma::raw_ascii);
    if (type == tanh){
        output->transform([](double val) {return std::tanh(val);});
    } else if(type == ReLU){
        output->transform([](double val) {return val > 0 ? val: 0 ;});
    } else if(type == sigmoid){
        output->transform([](double val) {return 1.0/(1.0+exp(-val));});
    }
//    output->save("convoLayer_output.dat", arma::raw_ascii);
}

void ConvolveLayer::updatePara(std::shared_ptr<arma::cube> delta_upper, double learningRate) {
//	Here we take the delta from upwards layer and calculate the new delta
// delta_upper has numofslices of depth of upper cube
    int halfSize_x = filterDim_x / 2;
    int halfSize_y = filterDim_y / 2;

// TODO 
// Later should consider case of sigmoid
    arma::cube delta;
    
    if (type == tanh) {
        arma::cube deriv = (1-(*output) % (*output));
        delta = (*delta_upper) % deriv;
    } else if (type == ReLU) {
        arma::ucube deriv = (*output) > 1e-15;
        delta = (*delta_upper) % deriv;
    } else if (type ==sigmoid){
        arma::cube deriv = (1-(*output)) % (*output);
        delta = (*delta_upper) % deriv;    
    } else if (type == linear){
        delta = *delta_upper;
    }
    
//    arma::cube delta = (*delta_upper) % deriv;
    MatArray<double>::Mat2DArray_ptr deltaW = MatArray<double>::build(numFilters, inputDim_z, filterDim_x, filterDim_y);
    MatArray<double>::fillZeros(deltaW);
    double sumtemp = 0.0;
    (*B) -= learningRate * delta;
    for (int filterIdx = 0; filterIdx < numFilters; filterIdx++) {
        for (int imIdx_z = 0; imIdx_z < inputDim_z ; imIdx_z++) {
            for (int m = 0; m < filterDim_x; m++) {
                for (int n = 0; n < filterDim_y; n++) {
                    for (int j = 0; j < outputDim_x; j++ ) {
                        for (int k = 0; k < outputDim_y; k++) {
                            int imIdx_x = j - m + halfSize_x;
                            int imIdx_y = k - n + halfSize_y;
//                            int imIdx_x = (m - halfSize) - j;
                            // delta has larger size than delta_up
                            if (imIdx_x >=0 && imIdx_x < inputDim_x &&
                                    imIdx_y >=0 && imIdx_y < inputDim_y) {
                                (*deltaW)[filterIdx][imIdx_z](m,n) +=
                                    delta(j,k,filterIdx) * (*input)(imIdx_x, imIdx_y, imIdx_z);
                              // here should satisfy imIdx_x + m = j  
                            }
                        }
                    }
                }
            }
        }
    }
//    input->print();
//    delta.print("delta");
     delta_out = std::make_shared<arma::cube>(inputDim_x, inputDim_y, inputDim_z, arma::fill::zeros);   
    // now calculate delta_out
        for (int imIdx_z = 0; imIdx_z < inputDim_z ; imIdx_z++) {
            for (int j = 0; j < inputDim_x; j++ ) {
                for (int k = 0; k < inputDim_y; k++) {
                    for (int filterIdx = 0; filterIdx < numFilters; filterIdx++) {                     
                        for (int m = 0; m < filterDim_x; m++) {
                            for (int n = 0; n < filterDim_y; n++) {                                
                            int imIdx_x = j + (m - halfSize_x);
                            int imIdx_y = k + (n - halfSize_y);
                            if (imIdx_x >=0 && imIdx_x < outputDim_x &&
                                    imIdx_y >=0 && imIdx_y < outputDim_y) {
                                (*delta_out)(j, k, imIdx_z) +=
                                    delta(imIdx_x, imIdx_y, filterIdx) * (*filters)[filterIdx][imIdx_z](m,n);
  //                                  std::cout << (*filters)[filterIdx][imIdx_z](m,n) << std::endl;
                            // here should satisfy imIdx = m + j        
                            }
                        }
                    }
                }
            }
        }
    }

    MatArray<double>::substract(filters, deltaW, learningRate);
//    MatArray<double>::save(deltaW,"analyGrad_conv");
//     delta_out->save("delta_out.dat",arma::raw_ascii);
}
