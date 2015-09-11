#include "NN_RLSolverBase.h"

using namespace ReinforcementLearning;
using namespace NeuralNet;
using namespace DeepLearning;
NN_RLSolverBase::NN_RLSolverBase(std::shared_ptr<BaseModel> m,
                         std::shared_ptr<Net> net0, 
                         std::shared_ptr<Trainer> trainer0, 
                         int Dim, DeepLearning::QLearningSolverParameter para):
                        RLSolverBase(m,Dim,para), net(net0), trainer(trainer0){
    netInputDim = stateDim + 1;
}

void NN_RLSolverBase::getMaxQ(const State& S, double* Q, int* action) {
    double maxQ;
    int a = 0;
    maxQ = -std::numeric_limits<double>::max();
    for (int j = 0; j < model->getNumActions(); j++) {
        double tempQ = this->calQ(S, j);
//        std::cout << tempQ << std::endl;
        if (maxQ < tempQ) {
            maxQ = tempQ;
            a = j;
        }
    }
    *Q = maxQ;
    *action = a;
    return;
}

