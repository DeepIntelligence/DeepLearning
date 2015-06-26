#include "PoolLayer.h"

PoolLayer::PoolLayer(int poolDim0_x, int poolDim0_y, Type type0) {
    poolDim_x = poolDim0_x;
    poolDim_y = poolDim0_y;
    type = type0;
}

void PoolLayer::setInputDim(int inputDim0_x, int inputDim0_y, int inputDim0_z){

    inputDim_x = inputDim0_x;
    inputDim_y = inputDim0_y;
    inputDim_z = inputDim0_z;
    inputSize = inputDim_x * inputDim_y * inputDim_z;
    outputDim_x = inputDim0_x / poolDim_x;
    outputDim_y = inputDim0_y / poolDim_y;
    outputDim_z = inputDim0_z;
    outputSize = outputDim_x * outputDim_y * outputDim_z;
}

void PoolLayer::activateUp(std::shared_ptr<arma::cube> input0) {
    input = input0;
    int maxIdx1, maxIdx2;
    int inputInstance = input->n_slices / inputDim_z;    
    output = std::make_shared<arma::cube>(outputDim_x,outputDim_y, outputDim_z*inputInstance,arma::fill::zeros);
    maxIdx_x = std::make_shared<arma::Cube<int>>(outputDim_x,outputDim_y, outputDim_z*inputInstance);
    maxIdx_y = std::make_shared<arma::Cube<int>>(outputDim_x,outputDim_y, outputDim_z*inputInstance);

    if (type == mean) {
        for (int d = 0; d < outputDim_z * inputInstance; d++) {
            for (int i = 0; i < outputDim_x; i++) {
                for (int j = 0; j < outputDim_y; j++) {
                    for (int m = i * poolDim_x; m < (i + 1) * poolDim_x; m++) {
                        for (int n = j * poolDim_y; n < (j + 1) * poolDim_y; n++) {
                            (*output)(i,j,d) += (*input)(m,n,d);
                        }
                    }
                }
            }
            (*output).slice(d) /= (1.0 * poolDim_x * poolDim_y);
        }
    } else if (type == max) {
        (*output).zeros();
        for (int d = 0; d < outputDim_z * inputInstance; d++) {
            for (int i = 0; i < outputDim_x; i++) {
                for (int j = 0; j < outputDim_y; j++) {
                    double maxtemp = 0.0;
                    maxIdx1 = 0;
                    maxIdx2 = 0;
                    for (int m = i * poolDim_x; m < (i + 1) * poolDim_x; m++) {
                        for (int n = j * poolDim_y; n < (j + 1) * poolDim_y; n++) {
                            if (maxtemp < (*input)(m,n,d) ) {
                                maxtemp = (*input)(m,n,d);
                                maxIdx1 = m;
                                maxIdx2 = n;
                            }
                        }
                    }
                    (*output)(i,j,d) = maxtemp;
                    (*maxIdx_x)(i,j,d) = maxIdx1;
                    (*maxIdx_y)(i,j,d) = maxIdx2;
                }
            }
        }
    }
}

void PoolLayer::upSampling(std::shared_ptr<arma::cube> delta_in) {
    int inputInstance = delta_in->n_slices / inputDim_z;
    delta_out = std::make_shared<arma::cube>(inputDim_x,inputDim_y, inputDim_z * inputInstance, arma::fill::zeros);
    if (type == mean) {
        for (int d = 0; d < inputDim_z * inputInstance; d++) {
            for (int i = 0; i < inputDim_x; i++) {
                for (int j = 0; j < inputDim_y; j++) {
                    (*delta_out)(i,j,d) = (*delta_in)(i/poolDim_x,j/poolDim_y,d);
                }
            }
        }
        (*delta_out) /= (1.0 * poolDim_x * poolDim_y);
    } else if(type == max) {
        for (int d = 0; d < outputDim_z * inputInstance; d++) {
            for (int i = 0; i < outputDim_x; i++) {
                for (int j = 0; j < outputDim_y; j++) {
                    (*delta_out)((*maxIdx_x)(i,j,d),(*maxIdx_y)(i,j,d),d) = (*delta_in)(i,j,d);             
                }
            }
        }
    }

}