#include "ConvolveLayer.h"


ConvolveLayer::ConvolveLayer(int numFilters0, int filterDim0_x, int filterDim0_y,
                             int stride0) {
    numFilters = numFilters0;
    filterDim_x = filterDim0_x;
    filterDim_y = filterDim0_y;
    stride = stride0;
}

void ConvolveLayer::setInputDim(int inputdim0_x, int inputDim0_y, int inputdim0_z){

    inputDim_x = inputDim_x;
    inputDim_y = inputDim_y;
    inputDim_z = inputDim_z;

    outputDim_x = inputDim_x / stride;
    outputDim_y = inputDim_y / stride;
    outputDim_z = inputDim_z;

    initializeWeight();
}

void ConvolveLayer::initializeWeight() {

    B = std::make_shared<arma::cube>(inputDim_x,inputDim_y, inputDim_z);
    filters = MatArray<double>::build(numFilters, , filterDim_x, filterDim_y);

}

void ConvolveLayer::activateUp() {
//  after convolution, the size of the image will usually shrink due to stride
    int halfSize_x = filterDim_x / 2;
    int halfSize_y = filterDim_y /2;

    for (int filterIdx = 0; filterIdx < numFilters; filterIdx++) {
        for (int imIdx_z = 0; imIdx_z < inputDim_z ; imIdx_z++) {
            for (int j = 0; j < inputDim_x; j+=stride ) {
                for (int k = 0; k < inputDim_y; k+=stride) {
                    for (int m = 0; m < filterDim_x; m++) {
                        for (int n = 0, n < filterDim_y; n++) {
                            int imIdx_x = j + m - halfSize_x;
                            int imIdx_y = k + n - halfSize_y;
                            if (imIdx_x >=0 && imIdx_x < inputDim_x && imIdx_y >=0 && imIdx_y < inputDim_y)
                                (*output)(j/stride,k/stride,filterIdx) += (*input)(imIdx_x, imIdx_y,imIdx_z) * filters[filterIdx][inDepth](m,n);
                        }
                    }
                    (*output)[filterIdx](j/stride,k/stride) += (*B)(filterIdx,inDepth);
                }
            }
        }
    }

    output->transform([](double val) {return tanh(val)});
}

void ConvolveLayer::updatePara(MatArray<double>::Mat1DArray delta_upper, const PoolLayer & pl) {
//	Here we take the delta from upwards layer and calculate the new delta
// delta_upper has numofslices of depth of upper cube
    int halfSize_x = filterDim_x / 2;
    int halfSize_y = filterDim_y / 2;

    delta = (*delta_upper) % (1-(*output)%(*output));
    (*B) -= delta;
    for (int filterIdx = 0; filterIdx < numFilters; filterIdx++) {
        for (int imIdx_z = 0; imIdx_z < inputDim_z ; inDepth++) {
            for (int m = 0; m < filterDim_x; m++) {
                for (int n = 0, n < filterDim_y; n++) {
                    for (int j = 0; j < outputDim_x; j++ ) {
                        for (int k = 0; k < outputDim_y; k++) {
                            int imIdx_x = j + m - halfSize;
                            int imIdx_y = k + n - halfSize;
                            // delta has larger size than delta_up
                            if (imIdx_x >=0 && imIdx_x < image_row &&
                                    imIdx_y >=0 && imIdx_y < image_col) {
                                (*deltaW)[filterIdx][imIdx_z](m,n) +=
                                    delta(j,k,filterIdx) * (*input)(j*stride-m, k*stride-n, filterIdx);

                                (*delta_out)(j*stride+m, k*stride+n, imIdx_z) +=
                                    delta(filterIdx, j, k) * (*deltaW)[filterIdx][imIdx_z](m,n);
                            }
                        }
                    }
                }
            }
        }
    }




}
