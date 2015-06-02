#include "PoolLayer.h"

PoolLayer::PoolLayer(int poolDim0_x, int poolDim0_y, Type type0) {
    poolDim_x = poolDim0_x;
    poolDim_y = poolDim0_y;
    type = type0;
}

void PoolLayer::setInput(std::shared<arma::cube> input0) {
    input = input0;

    inputDim_x = input->n_rows;
    inputDim_y = input->n_cols;
    inputDim_z = input->n_slices;

    outputDim_x = inputDim_x / poolDim0_x;
    outputDim_y = inputDim_y / poolDim0_y;
    outputDim_z = inputDim_z;

}

void PoolLayer::activateUp() {

    int maxIdx1, maxIdx2;
    output = std::make_shared<arma::cube>(outputDim_x,outputDim_y, outputDim_z);
    maxIdx_x = std::make_shared<arma::Cube<int>(inputDim_x,inputDim_y, inputDim_z);
    maxIdx_y = std::make_shared<arma::Cube<int>(inputDim_x,inputDim_y, inputDim_z);

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
            (*output).slice(d) /= (imageDim*imageDim);
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
                    (*output)[d](i,j) = maxtemp;
                    (*maxIdx_x)[d](i,j) = maxIdx1;
                    (*maxIdx_y)[d](i,j) = maxIdx2;
                }
            }
        }
    }
}

void PoolLayer::upSampling(std::shared<arma::cube> detla_in) {
    output = std::make_shared<arma::cube>(outputDim_x,outputDim_y, outputDim_z, arma::fill::zeros);
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
                    (*delta_out)(maxIdx_x(i,j,d),maxIdx_y(i,j,d),d) = (*delta_in)(i,j,d);
                }
            }
        }
    }

}