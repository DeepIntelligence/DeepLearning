#include "NN_RLSolverSimple.h"

using namespace ReinforcementLearning;
using namespace NeuralNet;
NN_RLSolverSimple::NN_RLSolverSimple(std::shared_ptr<BaseModel> m,
                         std::shared_ptr<Net> net0, 
                         std::shared_ptr<Trainer> trainer0, 
                         int Dim, DeepLearning::QLearningSolverParameter para):
                         NN_RLSolverBase(m,net0,trainer0,Dim,para){
    this->setNormalizationConst();
}

void NN_RLSolverSimple::setNormalizationConst(){
    state_norm.resize(stateDim+1);
    state_norm[0] = M_PI;
    state_norm[1] = 20.0;
    state_norm[2] = model->getNumActions()-1;
}


double NN_RLSolverSimple::getRewards(const State &newS) const{    
    if (this->terminate(newS)){
        return -1.0;
    } else {
        return 0.0;
    }
}
bool NN_RLSolverSimple::terminate(const State& S) const {
    return (S[0] < - 0.5* M_PI || S[0] > 0.5 * M_PI);

}