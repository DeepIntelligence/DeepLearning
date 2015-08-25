#include "NN_RLSolverMLP.h"

using namespace ReinforcementLearning;
using namespace NeuralNet;
NN_RLSolverMLP::NN_RLSolverMLP(std::shared_ptr<BaseModel> m,
                         std::shared_ptr<Net> net0, 
                         std::shared_ptr<Trainer> trainer0, 
                         int Dim, DeepLearning::QLearningSolverParameter para):
                         NN_RLSolverBase(m,net0,trainer0,Dim,para){
    this->setNormalizationConst();
}

void NN_RLSolverMLP::setNormalizationConst(){
    state_norm.resize(stateDim+1);
    state_norm[0] = M_PI;
    state_norm[1] = 20.0;
    state_norm[2] = model->getNumActions()-1;
}

double NN_RLSolverMLP::calQFromNet(std::shared_ptr<arma::mat> input){
	net->setTrainingSamples(inputTemp, nullptr);
	net->forward();
	double tempQ = arma::as_scalar(*(net->netOutput())); 
	return tempQ;
}
void NN_RLSolverMLP::train(){
    std::shared_ptr<arma::mat> trainingSampleX(new arma::mat);
    std::shared_ptr<arma::mat> trainingSampleY(new arma::mat);
    int maxIter = trainingPara.numtrainingepisodes();
    for (int iter = 0; iter < maxIter; iter++){
        std::cout << "RLsolver iteration: " << iter << std::endl;
        this->generateExperience();
        if (iter > 20) {
            this->generateTrainingSample(trainingSampleX, trainingSampleY);
//            trainingSampleX->print("X");
//            trainingSampleY->print("Y");
            trainer->setTrainingSamples(trainingSampleX, trainingSampleY);
            trainer->train();
        }
    }   
}

void NN_RLSolverMLP::generateTrainingSample(std::shared_ptr<arma::mat> trainingX, std::shared_ptr<arma::mat> trainingY){
    trainingX->set_size(netInputDim, experienceSet.size());
    trainingY->set_size(1, experienceSet.size());
    double maxQ;
    int action;
    std::shared_ptr<arma::mat> inputTemp(new arma::mat(netInputDim, 1));
    for (int i = 0; i < this->experienceSet.size(); i++) {
        this->getMaxQ(experienceSet[i].newState,&maxQ,&action);
        double targetQ = experienceSet[i].reward +  trainingPara.discount()*maxQ;;
        for ( int k = 0; k < this->stateDim; k++)
            inputTemp->at(k) =  experienceSet[i].oldState[k] / this->state_norm[k];
        inputTemp->at(stateDim) = experienceSet[i].action / state_norm[stateDim];
        
        trainingX->col(i) = *inputTemp;
        trainingY->at(i) = targetQ;
    }
}

void NN_RLSolverMLP::generateExperience(){
    double maxQ;
    int action;
    double epi = trainingPara.epsilon();
    arma::mat outputTemp(1,1);
    std::shared_ptr<arma::mat> inputTemp(new arma::mat(netInputDim, 1));
    model->createInitialState();
    int i;
    for(i = 0; i < trainingPara.episodelength(); i++){
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
}
double NN_RLSolverMLP::getRewards(const State &newS) const{    
    if (this->terminate(newS)){
        return -1.0;
    } else {
        return 0.0;
    }
}
bool NN_RLSolverMLP::terminate(const State& S) const {
    return (S[0] < - 0.5* M_PI || S[0] > 0.5 * M_PI);

}
