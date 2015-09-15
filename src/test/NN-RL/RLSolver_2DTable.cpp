#include "RLSolver_2DTable.h"

using namespace ReinforcementLearning;

RLSolver_2DTable::RLSolver_2DTable(std::shared_ptr<BaseModel> m, int Dim, 
        DeepLearning::QLearningSolverParameter para, int n_row0, int n_col0, 
        double dx, double dy, double min_x, double min_y):
    RLSolverBase(m,Dim,para){
    
    n_rows = n_row0;
    n_cols = n_col0;    
    dx1 = dx;
    dx2 = dy;
    minx1 = min_x;
    minx2 = min_y;
    numActions = model->getNumActions();
    QTable.zeros(n_rows, n_cols, numActions);
    count.zeros(n_rows, n_cols);
}

void RLSolver_2DTable::writeTrajectory(int iter, std::ostream &os, int action, State state, double reward) const {
    os << iter << "\t";
    for (int s = 0; s < this->stateDim; s++) {
        os << state[s] << "\t";
    }
    os << action << "\t";
    os << reward << std::endl;
}

void RLSolver_2DTable::train() {
    int iter;
    double maxQ, reward;
    int action;
    double epi = 0.95;
    int maxIter = trainingPara.numtrainingepisodes();
    int epiLength = trainingPara.episodelength();
    int controlFreq = 1;
    int trajOutputFreq = 10;
    int episodeOutputFreq = 10;
    bool experienceReplayFlag = false;
    for (int i = 0; i < maxIter; i++) {
        std::ofstream os;
        std::stringstream ss;
        ss << i;
        if ( (i+1)%episodeOutputFreq == 0) {
            os.open("traj/traj"+ss.str()+".dat");        
        }
        std::cout << "training Episodes " << i << std::endl;
        iter = 0;
        model->createInitialState();
        while (!model->terminate() && iter < epiLength) {
            State oldState = model->getCurrState();
            if (randChoice->nextDou() < epi){
                this->getMaxQ(oldState,&maxQ,&action);
            } else {
                action = randChoice->nextInt();
            }
            model->run(action, controlFreq);
            State newState = model->getCurrState();
            reward = model->getRewards();
            this->updateQ(Experience(oldState, newState, action, reward));
            if (experienceReplayFlag) {
                this->experienceVec.push_back(Experience(oldState, newState, action, reward));
            }
            
            if ((iter == 0 || (iter+1)%trajOutputFreq == 0) && (i+1)%episodeOutputFreq == 0) {
                this->writeTrajectory(iter, os, action, oldState, reward);
            }
             iter++;
        }
        // after an episode, do experience reply
        this->replayExperience();
        std::cout << "duration: " << iter << std::endl;
        if ((i + 1) % 100 == 0) {
            std::stringstream ss;        
            ss << i;
            this->outputQ("QTable_" + ss.str() + "iter");
        }
    }
        this->outputQ("QTableFinal");
        this->outputPolicy();

}

void RLSolver_2DTable::replayExperience(){
    for (auto exp = this->experienceVec.rbegin(); exp != this->experienceVec.rend(); ++exp){
        this->updateQ(*exp);
    }

}

void RLSolver_2DTable::test(){
// do some test
    int maxIter = trainingPara.numtrainingepisodes();
    int epiLength = trainingPara.episodelength();
    int iter;
    int action;
    double maxQ;
    for (int i = 0; i < 0.1*maxIter; i++) {
        std::cout << "testing Episodes " << i << std::endl;
        iter = 0;
        model->createInitialState();
        while (!model->terminate() && iter < epiLength) {
            State oldState = model->getCurrState();          
            this->getMaxQ(oldState,&maxQ,&action);
            model->run(action);
            State newState = model->getCurrState();
            double reward = model->getRewards();
            iter++;
        }  
         std::cout << "duration: " << iter << std::endl;
    }   
}

void RLSolver_2DTable::updateQ(Experience exp) {
    double learningRate = 0.1;
    int action;
    double maxQ;
    double discount = 0.95;
    this->getMaxQ(exp.newState, &maxQ, &action);
    std::pair<int, int> index0 = this->stateToIndex(exp.oldState);
    count(index0.first,index0.second) += 1;
    QTable(index0.first,index0.second,exp.action) += 
            learningRate * (exp.reward + discount * maxQ - QTable(index0.first, index0.second, exp.action));
}

void RLSolver_2DTable::getMaxQ(const State& S, double* maxQ, int* action) const{
    std::pair<int, int> index = this->stateToIndex(S);
    double max = -std::numeric_limits<double>::max();
    for (int i = 0; i < numActions; i++){
        if(max < QTable(index.first,index.second, i)){
            max = QTable(index.first,index.second, i);
            *action = i;
        }
    }
    *maxQ = max;
}

std::pair<int, int> RLSolver_2DTable::stateToIndex(const State& S) const {
    int idx1, idx2;
    idx1 = (int) ((S[0] - minx1)/dx1) + 1;
    idx2 = (int) ((S[1] - minx2)/dx2) + 1;
    return std::pair<int, int>(idx1,idx2);
}

void RLSolver_2DTable::outputQ(std::string filename) {
    
    for (int i = 0; i < numActions; ++i) {
        arma::mat temp = QTable.slice(i);
        std::stringstream ss;
        ss << i;
        temp.save(filename + ss.str() + ".dat", arma::raw_ascii);
    }
}

void RLSolver_2DTable::outputPolicy(){
    arma::Mat<int> actionMap(n_rows, n_cols, arma::fill::zeros);
    arma::mat QMap(n_rows, n_cols, arma::fill::ones);
    QMap *= -1;
    double maxQ;
    int action;
    for (int i = 0; i < n_rows; i++){
        for (int j = 0; j < n_cols; j++){
            maxQ = -1000;
            action = 0;
            for (int m = 0; m < numActions; m++){
                if(maxQ < QTable(i, j, m)){
                    maxQ = QTable(i, j, m);
                    action = m;
                }
            }
            actionMap(i,j) = action;
            QMap(i,j) = maxQ;
        }
    }
    actionMap.save("actionMap.dat", arma::raw_ascii);
    QMap.save("QMap.dat", arma::raw_ascii);
    count.save("count.dat", arma::raw_ascii);

}

