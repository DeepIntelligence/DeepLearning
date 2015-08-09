#include "NN_RLSolver.h"

using namespace ReinforcementLearning;
using namespace NeuralNet;
NN_RLSolver::NN_RLSolver(std::shared_ptr<BaseModel> m,
                         std::shared_ptr<Net> net0, 
                         std::shared_ptr<Trainer> trainer0, 
                         RL_TrainingPara tp, int Dim):
                        model(m), net(net0), trainer(trainer0), trainingPara(tp), stateDim(Dim){
    netInputDim = stateDim + 1;
    randChoice = std::make_shared<RandomStream>(0, model->getNumActions()-1);
    this->setNormalizationConst();
}
void NN_RLSolver::train(){
    std::shared_ptr<arma::mat> trainingSampleX(new arma::mat);
    std::shared_ptr<arma::mat> trainingSampleY(new arma::mat);
    
    for (int iter = 0; iter < trainingPara.maxIter; iter++){
        std::cout << "RLsolver iteration: " << iter << std::endl;
        this->generateExperience();
        if (iter > 20) {
            this->generateTrainingSample(trainingSampleX, trainingSampleY);
//            trainingSampleX->print("X");
//            trainingSampleY->print("Y");
            trainer->setTrainingSamples(trainingSampleX, trainingSampleY);
            trainer->train();
        }
        double sum = 0.0;
        for (double n : durationVec) 
            sum += n;
        std::cout << "averageDuration: " << sum/(iter+1) << std::endl;
    }   
}
void NN_RLSolver::setNormalizationConst(){
    state_norm.resize(stateDim+1);
    state_norm[0] = M_PI;
    state_norm[1] = 20.0;
    state_norm[2] = model->getNumActions()-1;
}

void NN_RLSolver::getMaxQ(const State& S, double* Q, int* action) {
    double maxQ;
    int a = 0;
    std::shared_ptr<arma::mat> inputTemp(new arma::mat(netInputDim, 1));
    maxQ = -std::numeric_limits<double>::max();
    for (int k = 0; k < this->stateDim; k++)
        inputTemp->at(k) = S[k] / this->state_norm[k];
    for (int j = 0; j < model->getNumActions(); j++) {
        inputTemp->at(stateDim) = j / state_norm[stateDim];
        net->setTrainingSamples(inputTemp, nullptr);
        net->forward();
        double tempQ = arma::as_scalar(*(net->netOutput()));
        if (maxQ < tempQ) {
            maxQ = tempQ;
            a = j;
        }
    }
    *Q = maxQ;
    *action = a;
    return;
}

void NN_RLSolver::generateTrainingSample(std::shared_ptr<arma::mat> trainingX, std::shared_ptr<arma::mat> trainingY){
    trainingX->set_size(netInputDim, experienceSet.size());
    trainingY->set_size(1, experienceSet.size());
    double maxQ;
    int action;
    std::shared_ptr<arma::mat> inputTemp(new arma::mat(netInputDim, 1));
    for (int i = 0; i < this->experienceSet.size(); i++) {
        this->getMaxQ(experienceSet[i].newState,&maxQ,&action);
        double targetQ = experienceSet[i].reward +  trainingPara.discount*maxQ;;
        for ( int k = 0; k < this->stateDim; k++)
            inputTemp->at(k) =  experienceSet[i].oldState[k] / this->state_norm[k];
        inputTemp->at(stateDim) = experienceSet[i].action / state_norm[stateDim];
        
        trainingX->col(i) = *inputTemp;
        trainingY->at(i) = targetQ;
    }
}

void NN_RLSolver::generateExperience(){
    double maxQ;
    int action;
    double epi = 0.95;
    arma::mat outputTemp(1,1);
    std::shared_ptr<arma::mat> inputTemp(new arma::mat(netInputDim, 1));
    model->createInitialState();
    int i;
    for(i = 0; i < trainingPara.trainingSampleSize; i++){
        if( this->terminate(model->getCurrState()) ) {
            break;
        }
        
        State oldState = model->getCurrState();
        if (randChoice->nextDou()< epi){
            this->getMaxQ(oldState, &maxQ, &action);
        } else {
            action = randChoice->nextInt();
        }
            model->run(action);
            State currState = model->getCurrState();
            double r = this->getRewards(currState);
            oldState.shrink_to_fit();
            currState.shrink_to_fit();
            this->experienceSet.push_back(Experience(oldState,currState, action, r));
    }
     std::cout << "duration:" << i << std::endl;
     durationVec.push_back(i);
}
void NN_RLSolver::generatePolicy() const{

}

double NN_RLSolver::getRewards(const State &newS) const{    
    if (this->terminate(newS)){
        return -1.0;
    } else {
        return 0.0;
    }
}
bool NN_RLSolver::terminate(const State& S) const {
    return (S[0] < - 0.5* M_PI || S[0] > 0.5 * M_PI);

}