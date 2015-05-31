#include "PoolLayer.h"



void PoolLayer::activateUp(){
	
  int imageDim = (*inputX)[0].n_cols;	
  int outputDim = imageDim / poolDim;
  int numSlices = inputX->size();
  int maxIdx1, maxIdx2;
  outputX = MatArray<double>::build(numSlices,outputDim,outputDim);
  maxIdx_x = MatArray<int>::build(numSlices,outputDim,outputDim);
  maxIdx_y = MatArray<int>::build(numSlices,outputDim,outputDim);
  
  if (type == mean){	
	  for (int d = 0; d < numSlices; d++){
		for (int i = 0; i < outputDim; i++){
                    for (int j = 0; j < outputDim; j++){
                        for (int m = i * poolDim; m < (i + 1) * poolDim; m++){
                            for (int n = j * poolDim; n < (j + 1) * poolDim; n++){
                                (*outputX)[d](i,j) += (*inputX)[d](m,n);
                                }
                            } 				
			}
		}
		 (*outputX)[d] /= (imageDim*imageDim); 
	  }
  } else if (type == max){
	  
	  for (int d = 0; d < numSlices; d++){
              (*outputX)[d].zeros();
		 for (int i = 0; i < outputDim; i++){
			for (int j = 0; j < outputDim; j++){
                            double maxtemp = 0.0;
                            for (int m = i * poolDim; m < (i + 1) * poolDim; m++){
                                for (int n = j * poolDim; n < (j + 1) * poolDim; n++){
                                    if (maxtemp < (*inputX)[d](m,n) ) {
                                        maxtemp = (*inputX)[d](m,n);
                                        maxIdx1 = m;
                                        maxIdx2 = n;
                                    }
                                }
                            }
			   (*outputX)[d](i,j) = maxtemp;
                           (*maxIdx_x)[d](i,j) = maxIdx1;
                           (*maxIdx_y)[d](i,j) = maxIdx2;
			}
		 }
	  }
  }
}