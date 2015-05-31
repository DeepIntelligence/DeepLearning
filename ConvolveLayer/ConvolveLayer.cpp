#include "ConvolveLayer.h"


ConvolveLayer::ConvolveLayer(int numFilters0, MatArray<double>::Mat1DArray input0){

    input = input0;
    inputDepth = input.size()
    numFilters = numFilters0;
    B = arma::mat(numFilters, inputDepth);	
}

ConvolveLayer::activateUp(){
	
        int inputDepth = input.size()
        int F = receptiveField;
        int S = stride;
        
	int halfSize = filterSize / 2;
        int image_row = input[inDepth].n_rows;
        int image_col = input[inDepth].n_cols;        
        
	for (int filterIdx = 0; filterIdx < numFilters; filterIdx++){		
		for (int inDepth = 0; inDepth < inputDepth ; inDepth++){
			for (int j = 0; j < image_row; j+=stride ){
				for (int k = 0; k < image_col; k+=stride){
					for (int m = 0; m < filterDim; m++){
						for (int n = 0, n < filterDim; n++){					
							int imxIdx = j + m - halfSize;
							int imyIdy = k + n - halfSize;
							if (imxIdx >=0 && imxIdx < image_row && imyIdy >=0 && imyIdy < image_col)
								output[filterIdx](j/stride,k/stride) += input[inDepth](imxIdx, imyIdy) * filters[filterIdx][inDepth](m,n);
						}
					};
					output[filterIdx](j/stride,k/stride) += B(filterIdx,inDepth);
				}
			}
		}
	}
		

} 

ConvolveLayer::updatePara(MatArray<double>::Mat1DArray delta_upper){
//	Here we take the delta from upwards layer and calculate the new delta
// delta_upper has numofslices of depth of upper cube

		for (int i = 0; i < numFilters; i++){
			for (int j = 0 ; j < K1; j++){
				for (int k = 0; k < numFilters; k++){
					for (int l = 0; l < )
					
					
				}		
				
				
			}
			
			
			
		}
	
	
	
	
	
}
