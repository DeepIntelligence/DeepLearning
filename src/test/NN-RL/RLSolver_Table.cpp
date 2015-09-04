#include "RLSolver_Table.h"

using namespace ReinforcementLearning;

RLSolver_Table::RLSolver_Table(std::shared_ptr<BaseModel> m, int Dim, DeepLearning::QLearningSolverParameter para):
    RLSolverBase(m,Dim,para){
    n_rows = 40;
    n_cols = 30;
    
    dx1 = 2.0*M_PI / n_rows;
    dx2 = 20.0 / n_cols;
    minx1 = -1.0*M_PI;
    minx2 = -10.0;
    numActions = model->getNumActions();
    QTable.zeros(n_rows, n_cols, numActions);
    count.zeros(n_rows, n_cols);
}

void RLSolver_Table::train() {
    int iter;
    double maxQ;
    int action;
    double epi = 0.5;
	int maxIter = trainingPara.numtrainingepisodes();
	int epiLength = trainingPara.episodelength();
    for (int i = 0; i < maxIter; i++) {
        std::cout << "training Episodes " << i << std::endl;
        iter = 0;
        model->createInitialState();
        while (!this->terminate(model->getCurrState()) && iter < epiLength) {
            State oldState = model->getCurrState();
            if (randChoice->nextDou() < epi){
                this->getMaxQ(oldState,&maxQ,&action);
            } else {
                action = randChoice->nextInt();
            }
            model->run(action);
            State newState = model->getCurrState();
            double reward = this->getRewards(newState);
            this->updateQ(Experience(oldState, newState, action, reward));
            iter++;
        }
        std::cout << "duration: " << iter << std::endl;
        if ((i + 1) % 1000 == 0) {
            std::stringstream ss;        
            ss << i;
            this->outputQ("QTable_" + ss.str() + "iter");
        }
    }
        this->outputQ("QTableFinal");
        this->outputPolicy();
// do some test
    for (int i = 0; i < 0.1*maxIter; i++) {
        std::cout << "testing Episodes " << i << std::endl;
        iter = 0;
        model->createInitialState();
        while (!this->terminate(model->getCurrState()) && iter < epiLength) {
            State oldState = model->getCurrState();          
            this->getMaxQ(oldState,&maxQ,&action);
            model->run(action);
            State newState = model->getCurrState();
            double reward = this->getRewards(newState);
            iter++;
        }  
         std::cout << "duration: " << iter << std::endl;
    }   
}

void RLSolver_Table::updateQ(Experience exp) {
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

void RLSolver_Table::getMaxQ(const State& S, double* maxQ, int* action) const{
    std::pair<int, int> index = this->stateToIndex(S);
    double max = -std::numeric_limits<double>::max();
    *action = 2;
    for (int i = 0; i < numActions; i++){
        if(max < QTable(index.first,index.second, i)){
            max = QTable(index.first,index.second, i);
            *action = i;
        }
    }
    *maxQ = max;
}

std::pair<int, int> RLSolver_Table::stateToIndex(const State& S) const {
    int idx1, idx2;
    idx1 = (int) ((S[0] - minx1)/dx1) + 1;
    idx2 = (int) ((S[1] - minx2)/dx2) + 1;
    return std::pair<int, int>(idx1,idx2);
}

void RLSolver_Table::outputQ(std::string filename) {
    
    for (int i = 0; i < numActions; ++i) {
        arma::mat temp = QTable.slice(i);
        std::stringstream ss;
        ss << i;
        temp.save(filename + ss.str() + ".dat", arma::raw_ascii);
    }
}

void RLSolver_Table::outputPolicy(){
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

double RLSolver_Table::getRewards(const State &newS) const {
    if (this->terminate(newS)) {
        return -1.0;
    } else {
        return 0.0;
    }
}

bool RLSolver_Table::terminate(const State& S) const {
    return (S[0] < -0.5 * M_PI || S[0] > 0.5 * M_PI);
}
