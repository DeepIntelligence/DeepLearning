#include "Trainer.h"

using namespace NeuralNet;
using namespace DeepLearning;

std::vector<std::shared_ptr<arma::mat>> Trainer::getGradientFromNet() {
    return net->netGradients();
}

void Trainer::applyUpdatesToNet(std::vector<std::shared_ptr<arma::mat>> update) {
    net->applyUpdates(update);
}

std::shared_ptr<arma::mat> Trainer::predict(std::shared_ptr<arma::mat> X) {
    net->setTrainingSamples(X, nullptr);
    net->forward();
    return net->netOutput();
}

std::vector<std::shared_ptr<arma::mat>> Trainer::predict(std::vector<std::shared_ptr<arma::mat>> X) {
    std::vector<std::shared_ptr<arma::mat>> outputVec;
    for (int i = 0; i < X.size(); i++){
        net->setTrainingSamples(X[i], nullptr);
        net->forward();
        outputVec.push_back(net->netOutput());
    }
    return outputVec;
}

void Trainer::trainHelper(std::shared_ptr<arma::mat> X, std::shared_ptr<arma::mat> Y) {
    net->setTrainingSamples(X, Y);
    net->calGradient();
    errorTotal += net->getLoss();
    currUpdate = this->getGradientFromNet();
    this->calUpdates();
    this->applyUpdatesToNet(this->currUpdate);

}

void Trainer::printInfo() {
    if (trainingParameter.neuralnettrainingparameter().verbose()&&
            (((iter + 1) % trainingParameter.neuralnettrainingparameter().printinfofrequency() == 0)
            || iter == 0)) {
        std::cout << "iteration: " << iter << std::endl;
        std::cout << "errorTotal: " << errorTotal << std::endl;
        std::cout << "learningRate:" << learningRate << std::endl;
    }
}

void Trainer::printGradNorm() {
    if (trainingParameter.neuralnettrainingparameter().verbose() &&
            trainingParameter.neuralnettrainingparameter().showgradnorm()) {
        for (int i = 0; i < currUpdate.size(); i++) {
            double grad_norm = arma::norm(*(currUpdate[i]), 2);
            std::cout << "norm of gradients are:" << "\t";
            std::cout << grad_norm << "\t";
        }
        std::cout << std::endl;
        }
}

void Trainer_SGD::calUpdates() {
    double momentum = trainingParameter.neuralnettrainingparameter().momentum();
    for (int i = 0; i < currUpdate.size(); i++) {
        if (iter > 0) {
            *(currUpdate[i]) = momentum * (*(prevUpdate[i])) + (*(currUpdate[i])) * learningRate;
        } else {
            *(currUpdate[i]) = (*(currUpdate[i])) * learningRate;
        } 
        
        if (trainingParameter.neuralnettrainingparameter().clipflag()){
            double grad_norm = arma::norm(*(currUpdate[i]),2);
            double threshold = trainingParameter.neuralnettrainingparameter().clipthreshold();
            if (grad_norm > threshold){
                (*currUpdate[i]) *= threshold / grad_norm; 
            }
        }
        *(prevUpdate[i]) = *(currUpdate[i]);
    }
    this->printGradNorm();
}

void Trainer_SGD::train() {
    std::shared_ptr<arma::mat> subTrainingX;
    std::shared_ptr<arma::mat> subTrainingY;
    int size = trainingParameter.neuralnettrainingparameter().minibatchsize();
    if (size < 0 || size >= trainingX->n_cols) {
        size = trainingX->n_cols;
    }    
    for (iter = 0; iter < trainingParameter.neuralnettrainingparameter().nepoch(); iter++) {
        errorTotal = 0.0;
        learningRate = trainingParameter.neuralnettrainingparameter().learningrate() / size;    
        double decayRate = trainingParameter.neuralnettrainingparameter().decayrate();
        learningRate = learningRate * exp(-iter / decayRate);
    
        int ntimes = this->trainingX->n_cols / size;
            if (trainingParameter.neuralnettrainingparameter().rnnscanflag()){
                
                int RNNScanStep = trainingParameter.neuralnettrainingparameter().rnnscanstep();
                int RNNTruncateLength = trainingParameter.neuralnettrainingparameter().rnntruncatelength();
                learningRate *= size / RNNTruncateLength;
                ntimes = (this->trainingX->n_cols - RNNTruncateLength) / RNNScanStep;
            }
        for (int i = 0; i < ntimes; i++) {
            int startPoint, endPoint; 
            startPoint = i*size;
            endPoint = (i + 1) * size - 1;
            if (trainingParameter.neuralnettrainingparameter().rnnscanflag()){
                startPoint = i * trainingParameter.neuralnettrainingparameter().rnnscanstep();
                endPoint = startPoint + trainingParameter.neuralnettrainingparameter().rnntruncatelength() - 1;
            }
            subTrainingX = std::make_shared<arma::mat>(trainingX->cols(startPoint, endPoint));
            subTrainingY= std::make_shared<arma::mat>(trainingY->cols(startPoint, endPoint));           
            trainHelper(subTrainingX, subTrainingY);
        }
        printInfo();
    }
}

void Trainer_SGDRNN::train() {
    std::shared_ptr<arma::mat> subTrainingX;
    std::shared_ptr<arma::mat> subTrainingY;
    int size = trainingParameter.neuralnettrainingparameter().minibatchsize();
    if (size < 0 || size >= trainingXVec.size()) {
        size = trainingXVec.size();
    }

    int numInstance;
    
    for (iter = 0; iter < trainingParameter.neuralnettrainingparameter().nepoch(); iter++) {
        errorTotal = 0.0;
    
        int ntimes = this->trainingXVec.size() / size;
        for (int i = 0; i < ntimes; i++) {
                //this->gradientClear();
                numInstance = 0;
            for (int j = 0; j < size; j++){
                subTrainingX = trainingXVec[i*size + j];
                subTrainingY= trainingYVec[i*size + j];
                numInstance += subTrainingY->n_cols;
                net->setTrainingSamples(subTrainingX, subTrainingY);
                net->calGradient();
                errorTotal += net->getLoss();
                if (j == 0) {
                    this->currUpdate_accu = this->getGradientFromNet();
                } else {
                    currUpdate = this->getGradientFromNet();
                    this->gradientAccu(currUpdate);
                }
            }
                
            learningRate = trainingParameter.neuralnettrainingparameter().learningrate();
            double decayRate = trainingParameter.neuralnettrainingparameter().decayrate();
            learningRate = learningRate * exp(-iter / decayRate);
            learningRate /= numInstance;
            this->calUpdates();
            this->applyUpdatesToNet(currUpdate_accu);
        }
        

       printInfo();
    }
}

void Trainer_SGDRNN::calUpdates() {
    double momentum = trainingParameter.neuralnettrainingparameter().momentum();
    for (int i = 0; i < currUpdate_accu.size(); i++) {
        if (iter > 0) {
            *(currUpdate_accu[i]) = momentum * (*(prevUpdate_accu[i])) + (*(currUpdate_accu[i])) * learningRate;
        } else {
            *(currUpdate_accu[i]) = (*(currUpdate_accu[i])) * learningRate;
        }        
        *(prevUpdate_accu[i]) = *(currUpdate_accu[i]);
    }
}



void Trainer_RMSProp::calUpdates() {
    for (int i = 0; i < currUpdate.size(); i++) {
//        currUpdate[i]->print();
        for (int j = 0; j < (currUpdate[i])->n_elem; j++){
            squared_accu[i]->at(j) = this->rho * squared_accu[i]->at(j) + (1- this->rho) * currUpdate[i]->at(j) * currUpdate[i]->at(j);
//            std::cout << sqrt(squared_accu[i]->at(j) + this->epsilon) << std::endl;
            currUpdate[i]->at(j) = learningRate * currUpdate[i]->at(j) / sqrt(squared_accu[i]->at(j) + this->epsilon);
        }
    }
}

void Trainer_RMSProp::train() {
std::shared_ptr<arma::mat> subTrainingX;
    std::shared_ptr<arma::mat> subTrainingY;
    int size = trainingParameter.neuralnettrainingparameter().minibatchsize();
    if (size < 0 || size >= trainingX->n_cols) {
        size = trainingX->n_cols;
    }
    
    for (iter = 0; iter < trainingParameter.neuralnettrainingparameter().nepoch(); iter++) {
        errorTotal = 0.0;
        learningRate = trainingParameter.neuralnettrainingparameter().learningrate() / 
        trainingParameter.neuralnettrainingparameter().minibatchsize();    
        double decayRate = trainingParameter.neuralnettrainingparameter().decayrate();
        learningRate = learningRate * exp(-iter / decayRate);
    
        int ntimes = this->trainingX->n_cols / size;
        for (int i = 0; i < ntimes; i++) {
            subTrainingX = std::make_shared<arma::mat>(trainingX->cols(i*size, (i + 1) * size - 1));
            subTrainingY= std::make_shared<arma::mat>(trainingY->cols(i*size, (i + 1) * size - 1));           
            trainHelper(subTrainingX, subTrainingY);
        }
        printInfo();
    }
}

