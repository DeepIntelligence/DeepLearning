#include <algorithm>
#include "MultiLayerPerceptron.h"

MultiLayerPerceptron::MultiLayerPerceptron(int inputDim0, int outputDim0, int hiddenDim0, std::shared_ptr<arma::mat> trainingX0,
                                           std::shared_ptr<arma::mat> trainingY0, TrainingPara trainingPara0){

   
   inputDim = inputDim0;
   hiddenDim = hiddenDim0;
   outputDim = outputDim0;
   numLayers = 2;
   trainingX = trainingX0;
   trainingY = trainingY0;
   numInstance = trainingX->n_rows;
   trainingPara = trainingPara0;
   
   layers.push_back(BaseLayer(inputDim,hiddenDim,BaseLayer::sigmoid)); 
   layers.push_back(BaseLayer(hiddenDim,outputDim,BaseLayer::softmax));	
//   layers[0].W.print("layer 0  W");
//   layers[0].B.print("layer 0  B");
//   layers[1].W.print("layer 1  W");
//   layers[1].B.print("layer 1  B");
}


void MultiLayerPerceptron::train(){
  // Here I used stochastic gradient descent 
  // first do the forward propagate 
    trainingPara.print();
    int ntimes = numInstance / trainingPara.miniBatchSize;
    std::shared_ptr<arma::mat> subInputX, subInputY;
    double errorTotal;
    int size = trainingPara.miniBatchSize;
    double alpha = trainingPara.alpha / size;
    for(int epoch = 0; epoch < trainingPara.NEpoch; epoch++){
        std::cout << epoch << std::endl;
        errorTotal = 0.0;
        for (int i = 0; i < ntimes; i++){
// first do the propogation            
            subInputX = std::make_shared<arma::mat>(trainingX->rows(i*size,(i+1)*size-1));
            subInputY = std::make_shared<arma::mat>(trainingY->rows(i*size,(i+1)*size-1));
    
            layers[0].inputX = subInputX;
            layers[0].activateUp(subInputX);
            layers[1].inputX = layers[0].outputY;
            layers[1].activateUp(layers[1].inputX);
 //       layers[0].outputY->print("layer0 outputY:");
 //       layers[1].outputY->print("layer1 outputY:");
 //       std::shared_ptr<arma::mat> predictY = layers[1].outputY;
            arma::mat sigmoid_deriv2 = (*(layers[1].outputY)) % (1-*(layers[1].outputY));
            arma::mat delta2 = ((-*subInputY + *(layers[1].outputY)).st()) % sigmoid_deriv2.st();
            arma::mat grad1 =  delta2 * (*(layers[1].inputX));
            arma::vec deltaSum2 = arma::sum(delta2,1);

            arma::mat errortemp = (-*subInputY + *(layers[1].outputY)).st();
 //       errortemp.print();
            arma::vec error = arma::sum(errortemp,1);
//            error.print();
//        deltaSum2.print();
            errorTotal += arma::as_scalar(error.st() * error);
            *(layers[1].W) -= alpha*grad1;     
            *(layers[1].B) -= alpha*deltaSum2;
        
        
        // delta0 should have the dimension of hidden Dimension
            arma::mat sigmoid_deriv1 = (*(layers[0].outputY)) % (1-*(layers[0].outputY));
            arma::mat delta1 = ( (layers[1].W)->st() * delta2) % sigmoid_deriv1.st();
            arma::mat grad0 = delta1 * (*(layers[0].inputX));
            arma::vec deltaSum1 = arma::sum(delta1,1);
            *(layers[0].W) -=  alpha*grad0;     
            *(layers[0].B) -=  alpha*deltaSum1; 
     
        }
        std::cout << "error is: " << errorTotal << std::endl;
    }
    
//    layers[1].outputY->print("final prediction");
}
  
  //if(converge(W_aug_old,W_aug_new)) break;	  
  						 					 

void MultiLayerPerceptron::test(std::shared_ptr<arma::mat> trainingX,std::shared_ptr<arma::mat> trainingY){
    layers[0].inputX = trainingX;
    layers[0].activateUp(trainingX);
    layers[1].inputX = layers[0].outputY;
    layers[1].activateUp(layers[1].inputX);
    layers[1].outputY->save("testoutput.txt",arma::raw_ascii);

}
	
	
	

