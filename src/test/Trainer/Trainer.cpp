#include "Trainer.h"

using namespace NeuralNet;
using namespace DeepLearning;

std::vector<std::shared_ptr<arma::mat>> Trainer::getGradientFromNet() {
    return net->netGradients();
}

void Trainer::applyUpdatesToNet() {
    net->applyUpdates(currUpdate);
}

void Trainer_SGD::calUpdates() {
    double momentum = trainingParameter.neuralnettrainingparameter().momentum();
    for (int i = 0; i < currUpdate.size(); i++) {
        if (iter > 0) {
            *(currUpdate[i]) = momentum * (*(prevUpdate[i])) + (*(currUpdate[i])) * learningRate;
        } else {
            *(currUpdate[i]) = (*(currUpdate[i])) * learningRate;
        }        
        *(prevUpdate[i]) = *(currUpdate[i]);
    }
}

void Trainer_SGD::train() {
    std::shared_ptr<arma::mat> subTrainingX;
    std::shared_ptr<arma::mat> subTrainingY;
    int size = trainingParameter.neuralnettrainingparameter().minibatchsize();
    double errorTotal;
    
    for (iter = 0; iter < trainingParameter.neuralnettrainingparameter().nepoch(); iter++) {
        errorTotal = 0.0;
        std::cout << "iteration: " << iter << std::endl;
        learningRate = trainingParameter.neuralnettrainingparameter().learningrate() / 
        trainingParameter.neuralnettrainingparameter().minibatchsize();
        
        double decayRate = trainingParameter.neuralnettrainingparameter().decayrate();
        learningRate = learningRate * exp(-iter / decayRate);
    
        int ntimes = this->trainingX->n_cols / size;
        for (int i = 0; i < ntimes; i++) {
            subTrainingX = std::make_shared<arma::mat>(trainingX->cols(i*size, (i + 1) * size - 1));
            subTrainingY= std::make_shared<arma::mat>(trainingY->cols(i*size, (i + 1) * size - 1));           
            net->setTrainingSamples(subTrainingX, subTrainingY);
            currUpdate = net->calGradient();
            errorTotal += net->getLoss();
            this->getGradientFromNet();
            this->calUpdates();
            this->applyUpdatesToNet();
        }
        std::cout << "errorTotal: " << errorTotal << std::endl;
        std::cout << "learningRate:" << learningRate << std::endl;
    }
}

void Trainer_iRProp::calUpdates() {
#if 0    
    double momentum = trainingParameter.neuralnettrainingparameter().momentum();
    double eps = 1e-12;
    double sign;
    double deltaMax;
    double deltaMin;
    double eta_plus;
    double eta_minus;
    for (int i = 0; i < currUpdate.size(); i++) {
        int size = currUpdate[i]->n_elem;
        if (iter > 0) {
            for (int j = 0; j < size; j++) {
                if (currUpdate[i]->at(j)*prevUpdate[i]->at(j) > eps ) {
                    currDelta(j) = std::min(prevDelta(j)*eta_plus, deltaMax);
                    sign = 1;
                    prevDelta(j) = currDelta(j);
                } else if (currUpdate[i]->at(j)*prevUpdate[i]->at(j) < -eps) {
                    currDelta(j) = std::max(prevDelta(j)*eta_minus, deltaMin);
                    sign = -1;
                    prevDelta(j) = currDelta(j);
                    prevUpdate(j) = 0.0;
                } else {
                    sign = -1;
                }
                currUpdate(j) = sign * currDelta(j);
            }
            *(prevUpdate[i]) = *(currUpdate[i]);
            *(prevDelta[i]) = *(currUpdate[i]);
            
        } else {
        
        
        }
    }
#endif
}

void Trainer_iRProp::train() {
    std::shared_ptr<arma::mat> subTrainingX;
    std::shared_ptr<arma::mat> subTrainingY;
    int size = trainingParameter.neuralnettrainingparameter().minibatchsize();
    double errorTotal;
    
    for (iter = 0; iter < trainingParameter.neuralnettrainingparameter().nepoch(); iter++) {
        errorTotal = 0.0;
        std::cout << "iteration: " << iter << std::endl;
    
        int ntimes = this->trainingX->n_cols / size;
        for (int i = 0; i < ntimes; i++) {
            subTrainingX = std::make_shared<arma::mat>(trainingX->cols(i*size, (i + 1) * size - 1));
            subTrainingY= std::make_shared<arma::mat>(trainingY->cols(i*size, (i + 1) * size - 1));           
            net->setTrainingSamples(subTrainingX, subTrainingY);
            net->calGradient();
            errorTotal += net->getLoss();
            this->getGradientFromNet();
            this->calUpdates();
            this->applyUpdatesToNet();
        }
        std::cout << "errorTotal: " << errorTotal << std::endl;
        std::cout << "learningRate:" << learningRate << std::endl;
    }
}

