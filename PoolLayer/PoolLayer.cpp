#include "PoolLayer.h"

PoolLayer::PoolLayer(int poolDim0_x, int poolDim0_y, Type type0) {
    poolDim_x = poolDim0_x;
    poolDim_y = poolDim0_y;
    type = type0;
}

void PoolLayer::setInputDim(int inputdim0_x, int inputDim0_y, int inputdim0_z){

    inputDim_x = inputDim_x;
    inputDim_y = inputDim_y;
    inputDim_z = inputDim_z;

    outputDim_x = inputDim_x / poolDim_x;
    outputDim_y = inputDim_y / poolDim_y;
    outputDim_z = inputDim_z;

}

void PoolLayer::activateUp(std::shared_ptr<arma::cube> input0) {
    input = input0;
    int maxIdx1, maxIdx2;
    output = std::make_shared<arma::cube>(outputDim_x,outputDim_y, outputDim_z);
    maxIdx_x = std::make_shared<arma::Cube<int>>(inputDim_x,inputDim_y, inputDim_z);
    maxIdx_y = std::make_shared<arma::Cube<int>>(inputDim_x,inputDim_y, inputDim_z);

    if (type == mean) {
        for (int d = 0; d < inputDim_z; d++) {
            for (int i = 0; i < outputDim_x; i++) {
                for (int j = 0; j < outputDim_y; j++) {
                    for (int m = i * poolDim_x; m < (i + 1) * poolDim_x; m++) {
                        for (int n = j * poolDim_y; n < (j + 1) * poolDim_y; n++) {
                            (*output)(i,j,d) += (*input)(m,n,d);
                        }
                    }
                }
            }
            (*output).slice(d) /= (poolDim_x * poolDim_y);
        }
    } else if (type == max) {
        (*output).zeros();
        for (int d = 0; d < inputDim_z; d++) {
            for (int i = 0; i < inputDim_x; i++) {
                for (int j = 0; j < inputDim_y; j++) {
                    double maxtemp = 0.0;
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
    delta_out = std::make_shared<arma::cube>(outputDim_x,outputDim_y, outputDim_z, arma::fill::zeros);
    if (type == mean) {
        for (int d = 0; d < outputDim_z; d++) {
            for (int i = 0; i < outputDim_x; i++) {
                for (int j = 0; j < outputDim_y; j++) {
                    for (int ii = 0; ii < poolDim_x; ii+=poolDim_x) {
                        for (int jj = 0; jj < poolDim_y; jj+=poolDim_y) {
                            (*delta_out)(ii,jj,d) = (*delta_in)(i,j,d);
                        }
                    }
                }
            }
        }
        (*delta_out) /= (poolDim_x * poolDim_y);
    } else if(type == max) {
        for (int d = 0; d < outputDim_z; d++) {
            for (int i = 0; i < outputDim_x; i++) {
                for (int j = 0; j < outputDim_y; j++) {
                    (*delta_out)((*maxIdx_x)(i,j,d),(*maxIdx_y)(i,j,d),d) = (*delta_in)(i,j,d);
                }
            }
        }
    }

}