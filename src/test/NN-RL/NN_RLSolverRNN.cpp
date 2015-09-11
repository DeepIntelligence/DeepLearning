#include "NN_RLSolverRNN.h"

using namespace ReinforcementLearning;
using namespace NeuralNet;
NN_RLSolverRNN::NN_RLSolverRNN(std::shared_ptr<BaseModel> m,
                         std::shared_ptr<Net> net0, 
                         std::shared_ptr<Trainer> trainer0, 
                         int Dim, DeepLearning::QLearningSolverParameter para):
                         NN_RLSolverMLP(m,net0,trainer0,Dim,para){}

double NN_RLSolverRNN::calQ(const State& S, int action) const{
    std::shared_ptr<arma::mat> inputTemp(new arma::mat(netInputDim, 1));
    for (int k = 0; k < stateDim; k++)
        inputTemp->at(k) = S[k] / this->state_norm[k];
    inputTemp->at(stateDim) = action / state_norm[stateDim] - 0.5;
    net->setTrainingSamples(inputTemp, nullptr);
    arma::mat output = net->forwardInTime(inputTemp);
    double tempQ = arma::as_scalar(output);
    return tempQ;
}

void NN_RLSolverRNN::train(){
    std::vector<std::shared_ptr<arma::mat>> trainingXVec, trainingYVec;
    int maxIter = trainingPara.numtrainingepisodes();
    for (int iter = 0; iter < maxIter; iter++){
        std::cout << "RLsolver iteration: " << iter << std::endl;
//        this->experienceSet.clear();
        this->experienceSet.clear();
        this->net->resetNetState();
        this->generateExperience();
        this->experienceSeqVec.push_back(experienceSet);    
        this->generateTrainingSampleVec(trainingXVec, trainingYVec);        
        if (iter >= 100 && (iter+1)%1 == 0) {  
            int idx = (iter / 1);
            std::stringstream s;
            s << idx;            
            outputTraining(trainingXVec, "X" + s.str()+".dat");
            outputTraining(trainingYVec, "Y" + s.str()+".dat");            
            trainer->setTrainingSamples(trainingXVec, trainingYVec);
            trainer->resetWeight();
            trainer->train();

            std::vector<std::shared_ptr<arma::mat>> predict = trainer->predict(trainingXVec);
            outputTraining(predict,"prediction"  + s.str()+".dat");
        
        
        }
    }   
}

void NN_RLSolverRNN::generateTrainingSampleVec(std::vector<std::shared_ptr<arma::mat>>& trainingXVec,
                                            std::vector<std::shared_ptr<arma::mat>>& trainingYVec){
    
    trainingXVec.clear();
    trainingYVec.clear();
    double mean = 0.0;
    double count = 0.0;
    for (int ii = 0; ii < this->experienceSeqVec.size(); ii++) {
        std::shared_ptr<arma::mat> trainingX( new arma::mat(netInputDim, experienceSeqVec[ii].size()));
        std::shared_ptr<arma::mat> trainingY( new arma::mat(1, experienceSeqVec[ii].size()));
        double maxQ;
        int action;
        std::shared_ptr<arma::mat> inputTemp(new arma::mat(netInputDim, 1));
        net->resetNetState();
        for (int i = 0; i < this->experienceSeqVec[ii].size(); i++) {
            for (int k = 0; k < this->stateDim; k++)
                inputTemp->at(k) = experienceSeqVec[ii][i].oldState[k] / this->state_norm[k];
            inputTemp->at(stateDim) = experienceSeqVec[ii][i].action / state_norm[stateDim] - 0.5;
            net->setTrainingSamples(inputTemp, nullptr);
            net->forwardInTime(inputTemp);
            net->updateInternalState();
            this->getMaxQ(experienceSeqVec[ii][i].newState, &maxQ, &action);
            double targetQ = experienceSeqVec[ii][i].reward + trainingPara.discount() * maxQ;
            ;

            trainingX->col(i) = *inputTemp;
            trainingY->at(i) = targetQ;
        }
        trainingXVec.push_back(trainingX);
        mean += arma::sum(arma::sum(*trainingY));        
        count += trainingY->n_elem;
        trainingYVec.push_back(trainingY);
        std::cout << "duration:" << trainingX->n_cols << std::endl;
    }
    // demean
    mean /= count;
    for (int i = 0; i < trainingYVec.size(); i++){
        *(trainingYVec[i]) -= mean;
    }
//    trainingX->print("X");
//    trainingY->print("Y");
}

void NN_RLSolverRNN::setNormalizationConst(){
    state_norm.resize(stateDim+1);
    state_norm[0] = M_PI;
    state_norm[1] = 5.0;
    state_norm[2] = model->getNumActions()-1;
}

void NN_RLSolverRNN::generateExperience(){
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
        // now I need to update the internal state of the network
        std::shared_ptr<arma::mat> inputTemp(new arma::mat(netInputDim, 1));
        for (int k = 0; k < this->stateDim; k++)
            inputTemp->at(k) = oldState[k] / this->state_norm[k];
        inputTemp->at(stateDim) = action / state_norm[stateDim];
        net->setTrainingSamples(inputTemp, nullptr);
        net->forwardInTime(inputTemp);
        net->updateInternalState();
        oldState.shrink_to_fit();
        currState.shrink_to_fit();        
        experienceSet.push_back(Experience(oldState,currState, action, r));
            
    }
    
}

bool NN_RLSolverRNN::terminate(const State& S) const {
    if (S[0] < - 0.5* M_PI || S[0] > 0.5 * M_PI 
    		|| S[1] < -2.4 || S[1] > 2.4){
//        std::cout << S[0]/state_norm[0] <<"\t" << S[1]/state_norm[1] << std::endl;    
        return true;
    } else {
        return false;
    }

}

void NN_RLSolverRNN::test(){
    std::ofstream os;
    os.open("test.dat");
}

void NN_RLSolverRNN::outputTraining(std::vector<std::shared_ptr<arma::mat>> &X, std::string filename){
    std::ofstream os1;
    os1.open(filename);
    for (int i = 0; i < X.size(); i++){
        (X[i])->print(os1);
    }
    os1.close();
}