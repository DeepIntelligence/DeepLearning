#include "ConvolveLayer.h"


ConvolveLayer::ConvolveLayer(  ){
	
	
	
}

ConvolveLayer::activateUp(std::shared_ptr<arma::cube> inputVolume){
	
	int K1 = inputVolume->nslices;
        int F = receptiveField;
        int K2 = numFilters;
        int S = stride;
        
	int halfSize = filterSize / 2;
        
        
	for (int filterIdx = 0; filterIdx < numFilters; filterIdx++){
		
		for (int inputDepth = 0; inputDepth < K1; inputDepth++){
                    arma::mat image = inputVolume->slice(inputDepth)
			for (int j = 0; j < image.n_rows; j+=stride ){
				for (int k = 0; k < image.n_cols; k+=stride){
					for (int m = 0; m < filterDim; m++){
						for (int n = 0, n < filterDim; n++){					
							int imxIdx = j + m - halfSize;
							int imyIdy = k + n - halfSize;
							if (imxIdx >=0 && imxIdx < imageDim && imyIdy >=0 && imyIdy < imageDim)
								(*output)(j/stride,k/stride,filterIdx) += inputVolume(imxIdx, imyIdy, inputDepth) * filter(n,m,filterIdx);
						}
					};
					(*output)(j/stride,k/stride,filterIdx) += B(filterIdx);
				}
			}
		}
	}
		

} 

/*ConvolveLayer::activateUp(std::shared_ptr<arma::cube> input){
	
	int numImage = input->nslices;
	int halfSize = filterSize / 2;
	for (int i = 0; i < numImage; i++){
		arma::image = input->slice(i);
		
		for (int filterIdx = 0; filterIdx < numFilters; filterIdx++){
			for (int j = 0; j < image.n_rows; j+=stride ){
				for (int k = 0; k < image.n_cols; k+=stride){
					for (int m = 0; m < filterDim; m++){
						for (int n = 0, n < filterDim; n++){					
							int imxIdx = j + m - halfSize;
							int imyIdx = k + n - halfSize;
							if (imxIdx >=0 && imxIdx < imageDim && imyIdx >=0 && imyIdx < imageDim)
								(*output)(j,k,i) += image(imxIdx, imyIdx) * filter(m,n,filterIdx);
						}
					}
					(*output)(j,k,i) += B(filterIdx);
				}
			}
		}
	}
		

} */