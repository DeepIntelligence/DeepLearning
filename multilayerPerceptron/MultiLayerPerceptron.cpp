#include <algorithm>
#include "MultiLayerPerceptron.h"

MultiLayerPerceptron::MultiLayerPerceptron(int numLayers0, std::vector<int> dimensions0, std::shared_ptr<arma::mat> trainingX0,
        std::shared_ptr<arma::mat> trainingY0, TrainingPara trainingPara0) {

    numLayers = numLayers0;
    dimensions = dimensions0;
    trainingX = trainingX0;
    trainingY = trainingY0;
    numInstance = trainingX->n_cols;
    trainingPara = trainingPara0;
    testGrad = false;
    totalDim = 0;
    for (int i = 0; i < numLayers; i++){
        if (i == numLayers-1) {
            layers.push_back(BaseLayer(dimensions[i],dimensions[i+1],BaseLayer::softmax));
        }else{
            layers.push_back(BaseLayer(dimensions[i],dimensions[i+1],BaseLayer::sigmoid));
        }
        totalDim += layers[i].totalSize;
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
    double errorTotal, crossEntropy;
    int size = trainingPara.miniBatchSize;
    double alpha = trainingPara.alpha / size;
    for(int epoch = 0; epoch < trainingPara.NEpoch; epoch++) {
        std::cout << epoch << std::endl;
        errorTotal = 0.0;
        crossEntropy = 0.0;
        for (int i = 0; i < ntimes; i++) {
// first do the propogation            
            subInputX = std::make_shared<arma::mat>(trainingX->cols(i*size,(i+1)*size-1));
            subInputY = std::make_shared<arma::mat>(trainingY->cols(i*size,(i+1)*size-1));
            feedForward(subInputX);
            
            if (testGrad){
                calNumericGrad(subInputX, subInputY);
                feedForward(subInputX);
            }
            
            std::shared_ptr<arma::mat> delta(new arma::mat);
             //for delta: each column is the delta of a sample
            *delta = (-*subInputY + *outputY);            
                       
            backProp(delta);
            delta->transform([](double val){return val*val;});
            errorTotal += arma::sum(arma::sum(*delta)); 
//            outputY->print();
            outputY->transform([](double val){return std::log(val+1e-20);});
           arma::mat crossEntropy_temp = *(subInputY) %  *(outputY);
//    crossEntropy_temp.save("crossEntropy.dat",arma::raw_ascii);
            crossEntropy -= arma::sum(arma::sum((crossEntropy_temp)));
//            std::cout << crossEntropy << std::endl;
//             std::cout << errorTotal << std::endl;
        }
            std::cout << "error is: " << errorTotal << std::endl;
            std::cout << "cross entropy is: " << crossEntropy << std::endl;
        
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
            (*delta) = (*outputY) - (*subInputY);
            *delta = arma::sum(*delta,1);
            error = 0.5* arma::as_scalar((*delta).st() * (*delta));
            temp_left = error;
            (*(layers[0].W))(i,j) -= 2.0*eps;
            feedForward(subInput);
 //           outputY->transform([](double val){return log(val);});
            (*delta) = (*outputY) - (*subInputY);
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


void MultiLayerPerceptron::calGrad(std::shared_ptr<arma::mat> delta_target){
    std::shared_ptr<arma::mat> delta_in = delta_target;
    for (int i = numLayers-1; i >= 0 ; i--){
        layers[i].calGrad(delta_in);
        delta_in = layers[i].delta_out;
    }
}

void MultiLayerPerceptron::test(std::shared_ptr<arma::mat> testingX,std::shared_ptr<arma::mat> testingY) {
        feedForward(testingX);
        std::shared_ptr<arma::mat> delta(new arma::mat);
             //for delta: each column is the delta of a sample
        *delta = (-*testingY + *outputY);  
        delta->transform([](double val){return val*val;});
//        arma::vec error = arma::sum(*delta,1);
        double errorTotal = arma::sum(arma::sum(*delta));                  
        outputY->transform([](double val){return std::log(val+1e-20);});
        arma::mat crossEntropy_temp = *(testingY) %  *(outputY);
//    crossEntropy_temp.save("crossEntropy.dat",arma::raw_ascii);
        double crossEntropy = - arma::sum(arma::sum((crossEntropy_temp))); 
        std::cout << "testing result:" << std::endl;
          std::cout << "error is: " << errorTotal << std::endl;
            std::cout << "cross entropy is: " << crossEntropy << std::endl;
}

void MultiLayerPerceptron::deVectoriseWeight(arma::vec &x){
        int startIdx = 0;
        int endIdx = 0;
//        std::shared_ptr<arma::vec> V(new arma::vec);
    for (int i = 0; i < numLayers; i++) {
        startIdx = endIdx;
        endIdx += layers[i].totalSize;
 //       *V = x.rows(startIdx,endIdx - 1);   
        layers[i].deVectoriseWeight(x.memptr(),startIdx);
        
    }
}

void MultiLayerPerceptron::vectoriseGrad(arma::vec &grad){
        int startIdx = 0;
        int endIdx = 0;
//        std::shared_ptr<arma::vec> V(new arma::vec);
    for (int i = 0; i < numLayers; i++) {
        startIdx = endIdx;
        endIdx += layers[i].totalSize;
 
        layers[i].vectoriseGrad(grad.memptr(), startIdx);
//        V->print();
//        grad.rows(startIdx,endIdx - 1) = *V;   
    }
}

void MultiLayerPerceptron::vectoriseWeight(arma::vec &x){
        int startIdx = 0;
        int endIdx = 0;
//        std::shared_ptr<arma::vec> V(new arma::vec);
    for (int i = 0; i < numLayers; i++) {
        startIdx = endIdx;
        endIdx += layers[i].totalSize;
        
        layers[i].vectoriseWeight(x.memptr(), startIdx);
//        x.rows(startIdx,endIdx - 1) = *V;
    }
}

MLPTrainer::MLPTrainer(MultiLayerPerceptron& MLP0):MLP(MLP0){
    dim = MLP.totalDim;  
    x_init = std::make_shared<arma::vec>(dim);
    MLP.vectoriseWeight(*x_init);
//    x_init->save("x_init.dat", arma::raw_ascii);
}
double MLPTrainer::operator ()(arma::vec& x, arma::vec& grad){

    grad.resize(MLP.totalDim);
//  first assign x to the weights and bias of all the layers    
    MLP.deVectoriseWeight(x);
    
    MLP.feedForward(MLP.trainingX);
    std::shared_ptr<arma::mat> delta(new arma::mat);
    //for delta: each column is the delta of a sample
    *delta = (-*(MLP.trainingY) + *(MLP.outputY));            
//    arma::vec error = arma::sum(*delta,1);
//  the error function we should have is the cross entropy 
//  since our gradient calcuated is assuming that we are using cross entropy
//    error.save("error.dat",arma::raw_ascii);
     MLP.outputY->transform([](double val){return std::log(val+1e-20);});

    arma::mat crossEntropy_temp = *(MLP.trainingY) %  *(MLP.outputY);
//    crossEntropy_temp.save("crossEntropy.dat",arma::raw_ascii);
    double crossEntropy = - arma::sum(arma::sum((crossEntropy_temp)));
//     std::cout << "cross entropy is: " << crossEntropy << std::endl;
    double errorTotal= 0.5 * arma::sum(arma::sum(delta->st() * (*delta)));            
    MLP.calGrad(delta);
    
    MLP.vectoriseGrad(grad);
    return crossEntropy;
}



