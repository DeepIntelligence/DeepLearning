#include "NN_RLSolverBase.h"

using namespace ReinforcementLearning;
using namespace NeuralNet;
using namespace DeepLearning;
NN_RLSolverBase::NN_RLSolverBase(std::shared_ptr<BaseModel> m,
                         std::shared_ptr<Net> net0, 
                         std::shared_ptr<Trainer> trainer0, 
                         int Dim, DeepLearning::QLearningSolverParameter para):
                        model(m), net(net0), trainer(trainer0), stateDim(Dim),
                        trainingPara(para){
    netInputDim = stateDim + 1;
    randChoice = std::make_shared<RandomStream>(0, model->getNumActions()-1);
}

void NN_RLSolverBase::getMaxQ(const State& S, double* Q, int* action) {
    double maxQ;
    int a = 0;
    std::shared_ptr<arma::mat> inputTemp(new arma::mat(netInputDim, 1));
    maxQ = -std::numeric_limits<double>::max();
    for (int k = 0; k < this->stateDim; k++)
        inputTemp->at(k) = S[k] / this->state_norm[k];
    for (int j = 0; j < model->getNumActions(); j++) {
        inputTemp->at(stateDim) = j / state_norm[stateDim];
        double tempQ = this->calQFromNet(inputTemp);
        if (maxQ < tempQ) {
            maxQ = tempQ;
            a = j;
        }
    }
    *Q = maxQ;
    *action = a;
    return;
}

