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

    B_size = outputSize;
    W_size = filterDim_x*filterDim_y*inputDim_z*numFilters;
    totalSize = B_size + W_size;
    initializeWeight();
}

void ConvolveLayer::initializeWeight() {
    B = std::make_shared<arma::cube>(outputDim_x,outputDim_y, outputDim_z,arma::fill::randu);
    B->transform([&](double val){return (val-0.5)/sqrt(outputSize);});
    filters = Tensor_4D::build(filterDim_x, filterDim_y,  inputDim_z, numFilters);
    (*filters).fill_randu();
    (*filters).transform([&](double val){return (val-0.5)/sqrt(outputSize);});
//    (*filters).fill_ones();
    grad_B = std::make_shared<arma::cube>(outputDim_x,outputDim_y, outputDim_z);
//    B->transform([&](double val){return (val-0.5)/sqrt(outputSize);});
    grad_W = Tensor_4D::build(filterDim_x, filterDim_y,  inputDim_z, numFilters);
        
    int KKD = filterDim_x * filterDim_y * inputDim_z;    
//    arma::mat filter2D(filters->getPtr(),KKD, numFilters, true);
    filters2D = std::make_shared<arma::mat>(filters->getPtr(),KKD, numFilters);
    grad_W2D = std::make_shared<arma::mat>();
    
                     
// here we try to initialize B and W in the continuous memory (wrong!) becasue now W and B is only safe to copy)
//    arma::vec v(totalSize,arma::fill::randu);
//    v.transform([&](double val){return (val-0.5)/sqrt(outputSize);});
//    filters = std::make_shared<Tensor_4D>(v.memptr(),W_size, filterDim_x, filterDim_y,  inputDim_z, numFilters);
//    double *v_ptr_offset = v.memptr() + W_size;
//    B = std::make_shared<arma::cube>(outputDim_x,outputDim_y, outputDim_z,arma::fill::randu);
    
    
}

void ConvolveLayer::convolve_naive(std::shared_ptr<arma::cube> input){
    int inputInstance = input->n_slices / inputDim_z;
    
    
    int halfSize_x = filterDim_x / 2;
    int halfSize_y = filterDim_y / 2;
    
   output = std::make_shared<arma::cube>(outputDim_x, outputDim_y, outputDim_z * inputInstance, arma::fill::zeros);
    for (int instance = 0; instance < inputInstance; instance++){
    for (int filterIdx = 0; filterIdx < numFilters; filterIdx++) {
        for (int imIdx_z = 0; imIdx_z < inputDim_z ; imIdx_z++) {
            for (int n = 0; n < filterDim_y; n++) {
                for (int m = 0; m < filterDim_x; m++) {                               
                        for (int k = 0; k < inputDim_y; k++) {
                            for (int j = 0; j < inputDim_x; j++ ) {
                            int imIdx_x = j - ( m - halfSize_x );
                            int imIdx_y = k - ( n - halfSize_y );
                            if (imIdx_x >=0 && imIdx_x < inputDim_x && imIdx_y >=0 && imIdx_y < inputDim_y)
                                (*output)(j,k,filterIdx + instance*numFilters) += 
                                        (*input)(imIdx_x, imIdx_y,imIdx_z + instance*inputDim_z) * (*filters)(m,n,imIdx_z,filterIdx);
                        }
                    }
                }
            }
        }
    }
    }
/* here is the edge augmentation method to avoid the if
    arma::cube outputTemp(inputDim_x+filterDim_x -1 , inputDim_y+filterDim_y -1, inputInstance*numFilters, arma::fill::zeros);
    for (int instance = 0; instance < inputInstance; instance++){
    for (int filterIdx = 0; filterIdx < numFilters; filterIdx++) {
        for (int imIdx_z = 0; imIdx_z < inputDim_z ; imIdx_z++) {
            for (int n = 0; n < filterDim_y; n++) {
                for (int m = 0; m < filterDim_x; m++) {           
                    
                        for (int k = 0; k < inputDim_y; k++) {
                            for (int j = 0; j < inputDim_x; j++ ) {
                            int imIdx_x = j + m;
                            int imIdx_y = k + n;
//                            if (imIdx_x >=0 && imIdx_x < inputDim_x && imIdx_y >=0 && imIdx_y < inputDim_y)
                                (outputTemp)(imIdx_x ,imIdx_y,filterIdx + instance*numFilters) += 
                                        (*input)(j ,k ,imIdx_z + instance*inputDim_z) * (*filters)(m,n,imIdx_z,filterIdx);
                        }
                    }
                }
            }
        }
    }
    }
    *output = outputTemp.tube(arma::span(halfSize_x,inputDim_x+halfSize_x -1), arma::span(halfSize_y, inputDim_y + halfSize_y -1));
//    std::cout << output->n_rows << std::endl;
*/
//   output->save("convolveoutput_naivemethod.dat",arma::raw_ascii);
}

void ConvolveLayer::activateUp(std::shared_ptr<arma::cube> subInput) {
//  after convolution, the size of the image will usually shrink due to stride
    
    int inputInstance = subInput->n_slices / inputDim_z;
    
    input = subInput;
    
    int halfSize_x = filterDim_x / 2;
    int halfSize_y = filterDim_y / 2;
//    
//    output->zeros();
//    input->save("convol_input.dat",arma::raw_ascii);
 //  convolve_naive(input);
//   filters2D->print();
    convolve_matrixMethod(input);
    
    
// change to following 1d array optimization    
    /*
    for (int instance = 0; instance < inputInstance; instance++){
    for (int filterIdx = 0; filterIdx < numFilters; filterIdx++) {
        for (int k = 0; k < outputDim_y; k++) {
            for (int j = 0; j < outputDim_x; j++ ) {            
                (*output)(j, k,filterIdx  + instance*numFilters) += (*B)(j, k,filterIdx);
                
            }
        }
    }
    }
    */
    double *output_ptr = output->memptr();
    double *B_ptr = B->memptr();
    for(int i = 0; i < inputInstance*outputSize; i++){
        *(output_ptr+i) += *(B_ptr+i%outputSize);
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
//   calGrad(delta_upper);
    calGrad_matrixMethod(delta_upper);
    (*B) -= learningRate * (*grad_B);
//    filters->substract(*grad_W, learningRate);
    *filters2D -= learningRate * (*grad_W2D);
//     delta_out->save("delta_out.dat",arma::raw_ascii);
}


void ConvolveLayer::calGrad(std::shared_ptr<arma::cube> delta_upper){
//	Here we take the delta from upwards layer and calculate the new delta
    int inputInstance = delta_upper->n_slices / numFilters;
// delta_upper has numofslices of depth of upper cube
    int halfSize_x = filterDim_x / 2;
    int halfSize_y = filterDim_y / 2;

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
    
    grad_W->fill_zeros();
//    grad_B = std::make_shared<arma::cube>(outputDim_x, outputDim_y, numFilters,arma::fill::zeros);
    grad_B->zeros();
    
// change to the following 1d array code
/*    
    for (int instance = 0; instance < inputInstance; instance++){
        for (int i = 0; i < outputDim_z; i++){
            grad_B->slice(i) += delta.slice(i + instance*numFilters);
        }
    }
*/
    double *B_ptr = grad_B->memptr();
    double *delta_ptr = delta.memptr();
    for(int i = 0; i < inputInstance*outputSize; i++){
        *(B_ptr+i % outputSize) += *(delta_ptr+i);
    }   
   
    for (int filterIdx = 0; filterIdx < numFilters; filterIdx++) {
        for (int imIdx_z = 0; imIdx_z < inputDim_z ; imIdx_z++) {
            for (int n = 0; n < filterDim_y; n++) {
                for (int m = 0; m < filterDim_x; m++) {                    
                  for (int instance = 0; instance < inputInstance; instance++){ 
                    for (int k = 0; k < outputDim_y; k++) {
                        for (int j = 0; j < outputDim_x; j++ ) {
                            int imIdx_x = j - m + halfSize_x;
                            int imIdx_y = k - n + halfSize_y;
//                            int imIdx_x = (m - halfSize) - j;
                            // delta has larger size than delta_up
                            if (imIdx_x >=0 && imIdx_x < inputDim_x &&
                                    imIdx_y >=0 && imIdx_y < inputDim_y) {
                                (*grad_W)(m, n, imIdx_z,filterIdx) +=
                                    delta(j, k,filterIdx + instance*numFilters) * (*input)(imIdx_x, imIdx_y, imIdx_z + instance*inputDim_z);
                              // here should satisfy imIdx_x + m = j  
                            }
                        }
                    }
                }
            }
        }
    }
}
//    input->print();
//    delta.print("delta");
     delta_out = std::make_shared<arma::cube>(inputDim_x, inputDim_y, inputDim_z*inputInstance, arma::fill::zeros);   
    // now calculate delta_out
    
      for (int filterIdx = 0; filterIdx < numFilters; filterIdx++) {
        for (int imIdx_z = 0; imIdx_z < inputDim_z ; imIdx_z++) {
            for (int n = 0; n < filterDim_y; n++) {
                for (int m = 0; m < filterDim_x; m++) {             
            for (int k = 0; k < inputDim_y; k++) {
                for (int j = 0; j < inputDim_x; j++ ) {
                    for (int instance = 0; instance < inputInstance; instance++){                     
                               
                            int imIdx_x = j + (m - halfSize_x);
                            int imIdx_y = k + (n - halfSize_y);
                            if (imIdx_x >=0 && imIdx_x < outputDim_x &&
                                    imIdx_y >=0 && imIdx_y < outputDim_y) {
                                (*delta_out)(j, k, imIdx_z + instance * inputDim_z) +=
                                    delta(imIdx_x, imIdx_y, filterIdx + instance*numFilters) * (*filters)(m,n,imIdx_z,filterIdx);
  //                                  std::cout << (*filters)[filterIdx][imIdx_z](m,n) << std::endl;
                            // here should satisfy imIdx = m + j        
                            }
                            }   
                        }
                    }
                }
            }
        }
    }
    grad_B->save("grad_B.dat", arma::raw_ascii);
    int KKD = filterDim_x * filterDim_y * inputDim_z;    
    arma::mat filter2D_temp(filters->getPtr(),KKD, numFilters);
    filter2D_temp.save("grad_W2D.dat", arma::raw_ascii); 
    delta_out->save("delta_out.dat", arma::raw_ascii);
}

void ConvolveLayer::calGrad_matrixMethod(std::shared_ptr<arma::cube> delta_upper){    
//  The idea of calculating gradient using matrix vector multiplication method from delta_upper (which is a cube) is that
//  consider we only have one image, then delta_upper has shape of (outputDim_x, outputDim_y, Numfilters)
//  the gradient of an element in the filter depends on which image elements are multiplied with the filter element    
//  1) transform the delta_upper to a matrix form (numFilters, outputDim_x*outputDim_y)      
//  2) since the input is matrix (KKD, inputDim_x*inputDim_y), and outputDim = inputDim
//  3) direct matrix product will give (KKD, numFilters) which is the gradient for the filter
//  4) therefore, there seems to be no need to store filters in 4D (numFilters, outputDim_x*outputDim_y)  

//  how to calculate the delta_out 
// 1) if we already shape the delta_upper to matrix from with size (numFilters, outputDim_x*outputDim_y)
// then delta_out = filter2D.st() * delta_upper2D, since filter2D has size of ( KKD , N ), the delta_out will
// has shape of (inputDim_x * inputDim_y, D)
    
//	Here we take the delta from upwards layer and calculate the new delta
    int inputInstance = delta_upper->n_slices / numFilters;
// delta_upper has numofslices of depth of upper cube
    int halfSize_x = filterDim_x / 2;
    int halfSize_y = filterDim_y / 2;

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
    
    
// delta_2D is transforming cube (outputDim_x, outputDim_y, numFilters * inputInstance)
// to (outputDim_x*outputDim_y*inputInstance, numFilters);    
    arma::mat delta_2D(outputDim_x*outputDim_y*inputInstance, numFilters);
    int width = outputDim_x * outputDim_y * inputInstance;
    int size = outputDim_x * outputDim_y;
    int row_idx;
    int col_idx, instance_idx;

    for (int j = 0; j < numFilters; j++){
        for (int i = 0; i < width; i++){
            instance_idx = i / size;
            row_idx = (i % size) % outputDim_x;
            col_idx = (i % size) / outputDim_x;
            delta_2D(i,j) = delta(row_idx, col_idx, j+instance_idx*numFilters);           
        }    
    }
    
    
    
// input2D has size (KKD, outputDim_x*outputDim_y*inputInstance))
// grad_w2d has size (KKD, N)    
    *grad_W2D = (*input2D) * (delta_2D);
        
    std::shared_ptr<arma::mat> delta_out_2D(new arma::mat);
  


    grad_B->zeros();

    double *B_ptr = grad_B->memptr();
    double *delta_ptr = delta.memptr();
    for(int i = 0; i < inputInstance * outputSize; i++){
        *(B_ptr+i % outputSize) += *(delta_ptr+i);
    }

 //   grad_B->save("grad_B_matrixmethod.dat", arma::raw_ascii);
 //   filters2D->save("grad_W2D_matrixmethod.dat", arma::raw_ascii);
// delta_2d has size ( outputDim^2*inputInstance, numFilters)    
// filters2d has size (KKD, NunFilters)   
// delta_out2D has size (KKD, outputDim^2*inputInstance)      
    *delta_out_2D =  (*filters2D) * delta_2D.st();   
    col2im(delta_out_2D, delta_out);  
    
//    delta_out->save("delta_out_matrixmethod.dat", arma::raw_ascii);
    
}



// vectorise grad is frequency used to pass out the gradient as a vector
void ConvolveLayer::vectoriseGrad(double *ptr, size_t offset){

    double *W_ptr = grad_W2D->memptr();
    double *B_ptr = grad_B->memptr();
    for (int i = 0; i < W_size; i++){
        *(ptr + offset) = *(W_ptr+i);
        offset++;
    }
    for (int i = 0; i < B_size; i++){
        *(ptr + offset) = *(B_ptr+i);
        offset++;
    }

}

void ConvolveLayer::vectoriseWeight(double *ptr, size_t offset){
    
    double *W_ptr = filters2D->memptr();
    double *B_ptr = B->memptr();
    for (int i = 0; i < W_size; i++){
        *(ptr + offset) = *(W_ptr+i);
        offset++;
    }
    for (int i = 0; i < B_size; i++){
        *(ptr + offset) = *(B_ptr+i);
        offset++;
    }
}

// devectorise weight is frequency used to pass out the gradient as a vector
void ConvolveLayer::deVectoriseWeight(double *ptr, size_t offset){
    
    double *W_ptr = filters2D->memptr();
    double *B_ptr = B->memptr();
    for (int i = 0; i < W_size; i++){
        *(W_ptr+i) = *(ptr + offset) ;
        offset++;
    }
    for (int i = 0; i < B_size; i++){
        *(B_ptr+i) = *(ptr + offset) ;
        offset++;
    }
}

void ConvolveLayer::convolve_matrixMethod(std::shared_ptr<arma::cube> input){
// first change to two d filter with size KKD * N
    int KKD = filterDim_x * filterDim_y * inputDim_z;
    int inputInstance = input->n_slices / inputDim_z;
//    arma::mat filter2D(filters->getPtr(),KKD, numFilters, true);

// transform the input to input2D
    im2col(input, input2D);
//    input->save("input3D.dat",arma::raw_ascii);
//    input2D->save("input2D.dat",arma::raw_ascii);
//    filters2D->save("filters2D.dat",arma::raw_ascii);
    
// input2D has size (KKD, outputDim_x*outputDim_y*inputInstance))   
// filters2D has size (KKD, numFilters)    
    arma::mat output2D = input2D->st() * (*filters2D);

//    output2D.save("convolveoutput2d.dat", arma::raw_ascii);
// output2D has size (outputDim_x*outputDim_y*inputInstance, NumFilters))
// we want output cube to have size (outputDim_x, outputDim_y, numFilters * inputInstance)
//    arma::vec output1D(output2D.memptr(), outputDim_x*outputDim_y*outputDim_z * inputInstance);
//    output1D.save("convolveoutput1d.dat", arma::raw_ascii);
    
//    arma::cube a(3,4,5,arma::fill::randn);
//    a.save("a.dat",arma::raw_ascii);
    
    output = std::make_shared<arma::cube> (outputDim_x, outputDim_y, outputDim_z * inputInstance); 
//    output->save("convolveoutput_matrixmethod1.dat", arma::raw_ascii);    
//    output->print("ouptutbefore");
    int width = outputDim_x * outputDim_y * inputInstance;
    int size = outputDim_x * outputDim_y;
    int row_idx;
    int col_idx, instance_idx;

    for (int j = 0; j < numFilters; j++){
        for (int i = 0; i < width; i++){
            instance_idx = i / size;
 //           instance_idx = i % inputInstance;
            row_idx = (i % size) % outputDim_x;
            col_idx = (i % size) / outputDim_x;
            (*output)(row_idx, col_idx, j+instance_idx*numFilters) = output2D(i,j);
 //           (*output)(row_idx, col_idx, j+instance_idx*numFilters) = output2D(i,j);
            
        }    
    }

//    output->save("convolveoutput_matrixmethod2.dat", arma::raw_ascii);
}

void ConvolveLayer::im2col(std::shared_ptr<arma::cube> input, std::shared_ptr<arma::mat>& output){
//    double *im_ptr = input->memptr();
//    double *out_ptr = outputImage->memptr();
    int inputInstance = input->n_slices / inputDim_z;
// delta_upper has numofslices of depth of upper cube
    int halfSize_x = filterDim_x / 2;
    int halfSize_y = filterDim_y / 2;
    
    
// input will be convert to (KKD, WH)    
    int KKD = filterDim_x * filterDim_y * inputDim_z;
    int KK = filterDim_x * filterDim_y;
    int inputDepth, H_idx, W_idx, input_row_idx, input_col_idx, instance_idx;
    int singleOutputImageWidth = inputDim_x * inputDim_y;
    int totalOutputImageWidth = singleOutputImageWidth * inputInstance;
    int filter_idx_x, filter_idx_y;
    
    output = std::make_shared<arma::mat>(KKD, totalOutputImageWidth);
    for( int output_col_idx = 0; output_col_idx < totalOutputImageWidth ; output_col_idx++){
        for (int output_row_idx = 0; output_row_idx < KKD; output_row_idx++){       
            
            inputDepth = output_row_idx / KK;
            
            instance_idx = output_col_idx / singleOutputImageWidth;
            
            H_idx = (output_col_idx % singleOutputImageWidth) % inputDim_x;
            W_idx = (output_col_idx % singleOutputImageWidth) / inputDim_x;
            
            filter_idx_x = (output_row_idx % KK) % filterDim_x;
            filter_idx_y = (output_row_idx % KK) / filterDim_x;

            input_row_idx = H_idx - (filter_idx_x - halfSize_x);
            input_col_idx = W_idx - (filter_idx_y - halfSize_y);
            
            if (input_row_idx >= 0 && input_row_idx < inputDim_x 
                    && input_col_idx >=0 && input_col_idx < inputDim_y){
            (*output)(output_row_idx, output_col_idx) = 
                    (*input)(input_row_idx, input_col_idx, inputDepth + instance_idx * inputDim_z);
            } else {
            (*output)(output_row_idx, output_col_idx) = 0.0;
            }
        
        }
    }    
//    output->save("output2Dimage.dat",arma::raw_ascii);
//    return outputImage;
}

void ConvolveLayer::col2im(std::shared_ptr<arma::mat> input, std::shared_ptr<arma::cube> &output){
// the input has size (KKD, outputDim_x*outputDim_y)
// we want to output to be a cube with size (inputDim_x, inputDim_y, inputDim_z))
    int singleInputImageWidth = inputDim_x * inputDim_y;
    int totalInputImageWidth = input->n_cols;
    
    int inputInstance = totalInputImageWidth / singleInputImageWidth;
    
    int halfSize_x = filterDim_x / 2;
    int halfSize_y = filterDim_y / 2;
    
    output = std::make_shared<arma::cube>(inputDim_x, inputDim_y, inputDim_z*inputInstance, arma::fill::zeros);   
    
// input has size (KKD, WH)    
    int KKD = filterDim_x * filterDim_y * inputDim_z;
    int KK = filterDim_x * filterDim_y;
    int outputDepth, H_idx, W_idx, output_row_idx, output_col_idx, instance_idx;
    int filter_idx_x, filter_idx_y;
 //   input->save("col2im_input.dat",arma::raw_ascii);
    for( int input_col_idx = 0; input_col_idx < totalInputImageWidth ; input_col_idx++){
        for (int input_row_idx = 0; input_row_idx < KKD; input_row_idx++){       
            
            outputDepth = input_row_idx / KK;
            instance_idx = input_col_idx / singleInputImageWidth;
            
            W_idx = (input_col_idx % singleInputImageWidth) / inputDim_x;
            H_idx = (input_col_idx % singleInputImageWidth) % inputDim_x;
            
            filter_idx_x = (input_row_idx % KK) % filterDim_x;
            filter_idx_y = (input_row_idx % KK) / filterDim_x;

            output_row_idx = H_idx - (filter_idx_x - halfSize_x);
            output_col_idx = W_idx - (filter_idx_y - halfSize_y);
// we should satisfy output_row_idx + filter_idx = H_idx
// because when we forward, output_row_idx contribute to H_idx             
            if (output_row_idx >= 0 && output_row_idx < inputDim_x 
                    && output_col_idx >=0 && output_col_idx < inputDim_y){
                (*output)(output_row_idx, output_col_idx, outputDepth + instance_idx * inputDim_z) +=
                        (*input)(input_row_idx, input_col_idx);
//                (*output)(H_idx, W_idx, outputDepth + instance_idx * inputDim_z) +=
//                        (*input)(input_row_idx, input_col_idx);
            } 
//            else{
//                std::cout << (*input)(input_row_idx, input_col_idx) << std::endl;
//            }
        
        }
    }    
}