#include <algorithm>
#include "MultiLayerPerceptron.h"

MultiLayerPerceptron::MultiLayerPerceptron(int numLayers0, std::vector<int> dimensions0, std::shared_ptr<arma::mat> trainingX0,
        std::shared_ptr<arma::mat> trainingY0, TrainingPara trainingPara0) {

    numLayers = numLayers0;
    dimensions = dimensions0;
    trainingX = trainingX0;
    trainingY = trainingY0;
    numInstance = trainingX->n_rows;
    trainingPara = trainingPara0;
    testGrad = false;
    for (int i = 0; i < numLayers; i++){
        if (i == numLayers-1) {
            layers.push_back(BaseLayer(dimensions[i],dimensions[i+1],BaseLayer::softmax));
        }else{
            layers.push_back(BaseLayer(dimensions[i],dimensions[i+1],BaseLayer::sigmoid));
        }
    }
//   layers[0].W.print("layer 0  W");
//   layers[0].B.print("layer 0  B");
//   layers[1].W.print("layer 1  W");
//   layers[1].B.print("layer 1  B");
}


void MultiLayerPerceptron::train() {
    // Here I used stochastic gradient descent
    // first do the forward propagate
    trainingPara.print();
    int ntimes = numInstance / trainingPara.miniBatchSize;
    std::shared_ptr<arma::mat> subInputX, subInputY;
    double errorTotal;
    int size = trainingPara.miniBatchSize;
    double alpha = trainingPara.alpha / size;
    for(int epoch = 0; epoch < trainingPara.NEpoch; epoch++) {
        std::cout << epoch << std::endl;
        errorTotal = 0.0;
        for (int i = 0; i < ntimes; i++) {
// first do the propogation            
            subInputX = std::make_shared<arma::mat>(trainingX->rows(i*size,(i+1)*size-1));
            subInputY = std::make_shared<arma::mat>(trainingY->rows(i*size,(i+1)*size-1));
            feedForward(subInputX);
            
            if (testGrad){
                calNumericGrad(subInputX, subInputY);
                feedForward(subInputX);
            }
            
            std::shared_ptr<arma::mat> delta(new arma::mat);
             //for delta: each column is the delta of a sample
            *delta = ((-*subInputY + *outputY).st());            
            arma::vec error = arma::sum(*delta,1);
            errorTotal += arma::as_scalar(error.st() * error);            
            backProp(delta);
           
        }
            std::cout << "error is: " << errorTotal << std::endl;
        
    }
 }

void MultiLayerPerceptron::feedForward(std::shared_ptr<arma::mat> subInput0){
    std::shared_ptr<arma::mat> subInput = subInput0;
    layers[0].inputX = subInput;
    for (int i = 0; i < numLayers; i++) {
        layers[i].activateUp(subInput);
        subInput = layers[i].outputY;
        if (i > 0) {
            layers[i].inputX = layers[i-1].outputY;
        }    
    }
    outputY = layers[numLayers-1].outputY;
}

void MultiLayerPerceptron::calNumericGrad(std::shared_ptr<arma::mat> subInput,std::shared_ptr<arma::mat> subInputY){
    std::shared_ptr<arma::mat> delta = std::make_shared<arma::mat>();
    int dim1 = layers[0].outputDim;
    int dim2 = layers[0].inputDim;
    double eps = 1e-9;
       
    arma::mat dW(dim1,dim2,arma::fill::zeros);

    double temp_left, temp_right;
    double error;
    
    for (int i = 0; i < dim1; i++){
        for (int j = 0; j < dim2; j++){
            (*(layers[0].W))(i,j) += eps;
            feedForward(subInput);
 //           outputY->transform([](double val){return log(val);});
            (*delta) = ((*outputY) - (*subInputY)).st();
            *delta = arma::sum(*delta,1);
            error = 0.5* arma::as_scalar((*delta).st() * (*delta));
            temp_left = error;
            (*(layers[0].W))(i,j) -= 2.0*eps;
            feedForward(subInput);
 //           outputY->transform([](double val){return log(val);});
            (*delta) = ((*outputY) - (*subInputY)).st();
            *delta = arma::sum(*delta,1);
            error = 0.5* arma::as_scalar((*delta).st() * (*delta));;
            temp_right = error;
            (*(layers[0].W))(i,j) += eps;
            dW(i,j) = (temp_left - temp_right) / 2.0 / eps;    
        }         
    }  
    dW.save("numGrad.dat",arma::raw_ascii);
}


void MultiLayerPerceptron::backProp(std::shared_ptr<arma::mat> delta_target){
    std::shared_ptr<arma::mat> delta_in = delta_target;
    double learningRate = trainingPara.alpha / trainingPara.miniBatchSize;
    for (int i = numLayers-1; i >= 0 ; i--){
        layers[i].updatePara(delta_in, learningRate );
        delta_in = layers[i].delta_out;
    }
}

void MultiLayerPerceptron::test(std::shared_ptr<arma::mat> trainingX,std::shared_ptr<arma::mat> trainingY) {
    layers[0].inputX = trainingX;
    layers[0].activateUp(trainingX);
    layers[1].inputX = layers[0].outputY;
    layers[1].activateUp(layers[1].inputX);
    layers[1].outputY->save("testoutput.txt",arma::raw_ascii);

}




