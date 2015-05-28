#include "PoolLayer.h"



void PoolLayer::activateUp(std::shared_ptr<arma::cube> inputX){
	
  int imageDim = inputX->n_cols;	
  int outputDim = imageDim / poolDim;
  int numSlices = inputX->n_slices;
  outputX = std::make_shared<arma::cube>(outputDim,outputDim,numSlices, arma::fill::zeros);
  if (type == mean){	
	  for (int d = 0; d < numSlices; d++){
		for (int i = 0; i < outputDim; i++){
                    for (int j = 0; j < outputDim; j++){
                        for (int m = i * poolDim; m < (i + 1) * poolDim; m++){
                            for (int n = j * poolDim; n < (j + 1) * poolDim; n++){
                                (*outputX)(i,j,d) += (*inputX)(m,n,d);
                                }
                            } 				
			}
		}
		 (*outputX).slice(d) /= (imageDim*imageDim); 
	  }
  } else if (type == max){
	  
	  for (int d = 0; d < numSlices; d++){
		 for (int i = 0; i < outputDim; i++){
			for (int j = 0; j < outputDim; j++){
                            double maxtemp = 0.0;
                            for (int m = i * poolDim; m < (i + 1) * poolDim; m++){
                                for (int n = j * poolDim; n < (j + 1) * poolDim; n++){
                                    if (maxtemp < (*inputX)(m,n,d) ) maxtemp = (*inputX)(m,n,d);
                                }
                            }
			   (*outputX)(i,j,d) = maxtemp; 				
			}
		 }
	  }
  }
}