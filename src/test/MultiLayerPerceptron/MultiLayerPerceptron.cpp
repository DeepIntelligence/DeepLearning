#include <algorithm>
#include "MultiLayerPerceptron.h"
using namespace NeuralNet;
using namespace DeepLearning;

MultiLayerPerceptron::MultiLayerPerceptron(NeuralNetParameter neuralNetPara0) {

    neuralNetPara = neuralNetPara0;
    numLayers = neuralNetPara.layerstruct_size();

    testGrad = false;
    totalDim = 0;
    for (int i = 0; i < numLayers; i++) {
        switch (neuralNetPara.layerstruct(i).activationtype()) {
            case LayerStructParameter_ActivationType_sigmoid:
                layers.push_back(BaseLayer(neuralNetPara.layerstruct(i).inputdim(),
                        neuralNetPara.layerstruct(i).outputdim(), BaseLayer::sigmoid));
                break;
            case LayerStructParameter_ActivationType_tanh:
                layers.push_back(BaseLayer(neuralNetPara.layerstruct(i).inputdim(),
                        neuralNetPara.layerstruct(i).outputdim(), BaseLayer::tanh));
                break;
            case LayerStructParameter_ActivationType_softmax:
                layers.push_back(BaseLayer(neuralNetPara.layerstruct(i).inputdim(),
                        neuralNetPara.layerstruct(i).outputdim(), BaseLayer::softmax));
                break;
            case LayerStructParameter_ActivationType_linear:
                layers.push_back(BaseLayer(neuralNetPara.layerstruct(i).inputdim(),
                        neuralNetPara.layerstruct(i).outputdim(), BaseLayer::linear));
                break;
            default:break;
        }
        totalDim += layers[i].totalSize;
    }

    for (int i = 0; i < numLayers; i++){
        netGradVector.push_back(this->layers[i].grad_W);
        netGradVector.push_back(this->layers[i].grad_B);
    }
}

void MultiLayerPerceptron::setTrainingSamples(std::shared_ptr<arma::mat> X, std::shared_ptr<arma::mat> Y) {
    trainingX = X;
    trainingY = Y;
    numInstance = trainingX->n_cols;
}

void MultiLayerPerceptron::calLoss(std::shared_ptr<arma::mat> delta){
    arma::mat errorMat = (*delta) % (*delta);
    double errorTotal = arma::sum(arma::sum(errorMat));
    this->error = errorTotal;
}

double MultiLayerPerceptron::getLoss(){
    return this->error;
}

void MultiLayerPerceptron::calGradient(){
     this->feedForward(this->trainingX);
 // now calculate propogate the error
    std::shared_ptr<arma::mat> delta(new arma::mat);
    *delta = (-*trainingY + *netOutput);
    backProp(delta);
     
    this->calLoss(delta);

}
#if 0
void MultiLayerPerceptron::train() {
    // Here I used stochastic gradient descent
    // first do the forward propagate
    //    trainingPara.print();
    int ntimes = numInstance / neuralNetPara.neuralnettrainingparameter().minibatchsize();
    std::shared_ptr<arma::mat> subInputX, subInputY;
    double errorTotal, crossEntropy;
    int size = neuralNetPara.neuralnettrainingparameter().minibatchsize();
    double learningRate = neuralNetPara.neuralnettrainingparameter().learningrate() / size;
    for (int epoch = 0; epoch < neuralNetPara.neuralnettrainingparameter().nepoch(); epoch++) {
        std::cout << epoch << std::endl;
        errorTotal = 0.0;
        crossEntropy = 0.0;
        for (int i = 0; i < ntimes; i++) {
            // first do the propogation            
            subInputX = std::make_shared<arma::mat>(trainingX->cols(i*size, (i + 1) * size - 1));
            subInputY = std::make_shared<arma::mat>(trainingY->cols(i*size, (i + 1) * size - 1));
            feedForward(subInputX);

            if (testGrad) {
                calNumericGrad(subInputX, subInputY);
                feedForward(subInputX);
            }

            std::shared_ptr<arma::mat> delta(new arma::mat);
            //for delta: each column is the delta of a sample
            *delta = (-*subInputY + *netOutput);
            backProp(delta, learningRate);

            delta->transform([](double val) {
                return val*val;
            });
            errorTotal += arma::sum(arma::sum(*delta));
            arma::mat netOutput_log(*netOutput);
            netOutput_log.transform([](double val) {
                return std::log(val + 1e-20);
            });
            arma::mat crossEntropy_temp = *(subInputY) % (netOutput_log);
            crossEntropy -= arma::sum(arma::sum((crossEntropy_temp)));
        }
        std::cout << "error is: " << errorTotal << std::endl;
        std::cout << "cross entropy is: " << crossEntropy << std::endl;

    }
}
#endif
void MultiLayerPerceptron::feedForward(std::shared_ptr<arma::mat> subInput0) {
    std::shared_ptr<arma::mat> subInput = subInput0;
    layers[0].input = subInput;
    for (int i = 0; i < numLayers; i++) {
        layers[i].activateUp(subInput);
        subInput = layers[i].output;
        if (i > 0) {
            layers[i].input = layers[i - 1].output;
        }
    }
    netOutput = layers[numLayers - 1].output;
}

void MultiLayerPerceptron::calNumericGrad(std::shared_ptr<arma::mat> subInput, std::shared_ptr<arma::mat> subInputY) {
    std::shared_ptr<arma::mat> delta = std::make_shared<arma::mat>();
    int dim1 = layers[0].outputDim;
    int dim2 = layers[0].inputDim;
    double eps = 1e-9;

    arma::mat dW(dim1, dim2, arma::fill::zeros);

    double temp_left, temp_right;
    double error;

    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            layers[0].W(i, j) += eps;
            feedForward(subInput);
            //           outputY->transform([](double val){return log(val);});
            (*delta) = (*netOutput) - (*subInputY);
            *delta = arma::sum(*delta, 1);
            error = 0.5 * arma::as_scalar((*delta).st() * (*delta));
            temp_left = error;
            layers[0].W(i, j) -= 2.0 * eps;
            feedForward(subInput);
            //           outputY->transform([](double val){return log(val);});
            (*delta) = (*netOutput) - (*subInputY);
            *delta = arma::sum(*delta, 1);
            error = 0.5 * arma::as_scalar((*delta).st() * (*delta));
            ;
            temp_right = error;
            layers[0].W(i, j) += eps;
            dW(i, j) = (temp_left - temp_right) / 2.0 / eps;
        }
    }
    dW.save("numGrad.dat", arma::raw_ascii);
}
void MultiLayerPerceptron::applyUpdates(std::vector<std::shared_ptr<arma::mat>> updates){
    for (int i = 0; i < numLayers; i++){
        this->layers[i].W -= *(updates[2*i]);
        this->layers[i].B -= *(updates[2*i+1]);
    }
}

std::vector<std::shared_ptr<arma::mat>> MultiLayerPerceptron::netGradients(){
    return netGradVector;
}


void MultiLayerPerceptron::backProp(std::shared_ptr<arma::mat> delta_target, double learningRate) {
    std::shared_ptr<arma::mat> delta_in = delta_target;
    for (int i = numLayers - 1; i >= 0; i--) {
        layers[i].updatePara(delta_in, learningRate);
        delta_in = layers[i].delta_out;
    }
}

void MultiLayerPerceptron::backProp(std::shared_ptr<arma::mat> delta_target) {
    std::shared_ptr<arma::mat> delta_in = delta_target;
    for (int i = numLayers - 1; i >= 0; i--) {
        layers[i].calGrad(delta_in);
        delta_in = layers[i].delta_out;
    }
}

void MultiLayerPerceptron::test(std::shared_ptr<arma::mat> testingX, std::shared_ptr<arma::mat> testingY) {
    feedForward(testingX);
    std::shared_ptr<arma::mat> delta(new arma::mat);
    //for delta: each column is the delta of a sample
    *delta = (-*testingY + *netOutput);
    delta->transform([](double val) {
        return val*val;
    });
    //        arma::vec error = arma::sum(*delta,1);
    double errorTotal = arma::sum(arma::sum(*delta));
    netOutput->transform([](double val) {
        return std::log(val + 1e-20);
    });
    arma::mat crossEntropy_temp = *(testingY) % *(netOutput);
    //    crossEntropy_temp.save("crossEntropy.dat",arma::raw_ascii);
    double crossEntropy = -arma::sum(arma::sum((crossEntropy_temp)));
    std::cout << "testing result:" << std::endl;
    std::cout << "error is: " << errorTotal << std::endl;
    std::cout << "cross entropy is: " << crossEntropy << std::endl;
}

void MultiLayerPerceptron::deVectoriseWeight(arma::vec &x) {
    int startIdx = 0;
    int endIdx = 0;
    //        std::shared_ptr<arma::vec> V(new arma::vec);
    for (int i = 0; i < numLayers; i++) {
        startIdx = endIdx;
        endIdx += layers[i].totalSize;
        //       *V = x.rows(startIdx,endIdx - 1);   
        layers[i].deVectoriseWeight(x.memptr(), startIdx);

    }
}

void MultiLayerPerceptron::vectoriseGrad(arma::vec &grad) {
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

void MultiLayerPerceptron::vectoriseWeight(arma::vec &x) {
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

void MultiLayerPerceptron::save(std::string filename) {
    char tag[10];
    for (int i = 0; i < this->numLayers; i++) {
        sprintf(tag, "%d", i);
        this->layers[i].save(filename + (std::string)tag);
    }
}
void MultiLayerPerceptron::load(std::string filename) {
    char tag[10];
    for (int i = 0; i < this->numLayers; i++) {
        sprintf(tag, "%d", i);
        this->layers[i].load(filename + (std::string)tag);
    }
}

/*
MLPTrainer::MLPTrainer(MultiLayerPerceptron& MLP0):MLP(MLP0){
    dim = MLP.totalDim;  
    x_init = std::make_shared<arma::vec>(dim);
    MLP.vectoriseWeight(*x_init);
//    x_init->save("x_init.dat", arma::raw_ascii);
}
double MLPTrainer::operator()(arma::vec& x, arma::vec& grad){

    grad.resize(MLP.totalDim);
//  first assign x to the weights and bias of all the layers    
    MLP.deVectoriseWeight(x);
    
    MLP.feedForward(MLP.trainingX);
    std::shared_ptr<arma::mat> delta(new arma::mat);
    //for delta: each column is the delta of a sample
 *delta = (-*(MLP.trainingY) + *(MLP.netOutput));            
//    arma::vec error = arma::sum(*delta,1);
//  the error function we should have is the cross entropy 
//  since our gradient calcuated is assuming that we are using cross entropy
//    error.save("error.dat",arma::raw_ascii);
     MLP.netOutput->transform([](double val){return std::log(val+1e-20);});

    arma::mat crossEntropy_temp = *(MLP.trainingY) %  *(MLP.netOutput);
//    crossEntropy_temp.save("crossEntropy.dat",arma::raw_ascii);
    double crossEntropy = - arma::sum(arma::sum((crossEntropy_temp)));
//     std::cout << "cross entropy is: " << crossEntropy << std::endl;
    double errorTotal= 0.5 * arma::sum(arma::sum(delta->st() * (*delta)));            
    MLP.calGrad(delta);
    
    MLP.vectoriseGrad(grad);
    return crossEntropy;
}
 */