#include "Trainer.h"

using namespace NeuralNet;
using namespace DeepLearning;

void Trainer::getGradientFromNet() {
    currUpdate = net->netGradients();
}

void Trainer::applyUpdatesToNet() {
    net->applyUpdates(currUpdate);
}

void Trainer_SGD::calUpdates() {
    double learningRate = trainingParameter.learningrate() / trainingParameter.minibatchsize();
    double momentum = trainingParameter.momentum();
    double decayRate = trainingParameter.decayrate();
    learningRate = learningRate * exp(-iter / decayRate);

    for (int i = 0; i < currUpdate.size(); i++) {
        *(currUpdate[i]) = momentum * (*(prevUpdate[i])) + (*(currUpdate[i])) * learningRate;
        *(prevUpdate[i]) = *(currUpdate[i]);
    }
}

void Trainer_SGD::train() {
    std::shared_ptr<arma::mat> subTrainingX;
    std::shared_ptr<arma::mat> subTrainingY;
    
    for (int i = 0; i < trainingParameter.nepoch(); i++) {        
        net->setTrainingSamples(subTrainingX, subTrainingY);
        net->calGradient();
        this->getGradientFromNet();
        this->calUpdates();
        this->applyUpdatesToNet();
    }
}
