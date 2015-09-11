#include "NN_RLSolverMultiMLP.h"

using namespace ReinforcementLearning;
using namespace NeuralNet;
NN_RLSolverMultiMLP::NN_RLSolverMultiMLP(std::shared_ptr<BaseModel> m,
                         std::vector<std::shared_ptr<Net>> net0, 
                         std::shared_ptr<Trainer> trainer0, 
                         int Dim, DeepLearning::QLearningSolverParameter para):
                         NN_RLSolverMLP(m,nullptr,trainer0,Dim,para){
    this->setNormalizationConst();
    this->netInputDim = stateDim;
    numActions = model->getNumActions();
    nets = net0; 
    // duplicate networks for each action
    for (int i = 0; i < numActions; i++){    	   
	trainingSampleXs.push_back(std::shared_ptr<arma::mat>(new arma::mat));
	trainingSampleYs.push_back(std::shared_ptr<arma::mat>(new arma::mat));
    }
    
    n_rows = 20;
    n_cols = 20;
    dx1 = 1.0*M_PI / n_rows;
    dx2 = 20.0 / n_cols;
    minx1 = -0.5*M_PI;
    minx2 = -1.0;
}

double NN_RLSolverMultiMLP::calQ(const State& S, int action) const {
    std::shared_ptr<arma::mat> inputTemp(new arma::mat(netInputDim, 1));
    for (int k = 0; k < stateDim; k++)
        inputTemp->at(k) = S[k] / this->state_norm[k];
    nets[action]->setTrainingSamples(inputTemp, nullptr);
    nets[action]->forward();
    double tempQ = arma::as_scalar(*(nets[action]->netOutput()));
    return tempQ;
}

void NN_RLSolverMultiMLP::train(){

    std::shared_ptr<arma::mat> prediction;
    int maxIter = trainingPara.numtrainingepisodes();
    for (int iter = 0; iter < maxIter; iter++){
        std::cout << "RLsolver iteration: " << iter << std::endl;
        this->generateExperience();
        if (iter > 2500 && (iter%20) == 0) {
            this->generateTrainingSample();
            for (int i = 0; i < numActions; i++) {
                
            	std::stringstream ss; 
            	ss << i;
            	trainingSampleXs[i]->save("X_net" + ss.str() + ".dat", arma::raw_ascii);
            	trainingSampleYs[i]->save("Y_net"+ ss.str() + ".dat", arma::raw_ascii);
            	trainer->setNet(nets[i]);
            	trainer->setTrainingSamples(trainingSampleXs[i], trainingSampleYs[i]);
                trainer->setLearningScalar(1.0 / (1 + 0.1*iter));
            	trainer->train();
            	prediction = trainer->predict(trainingSampleXs[i]);
                prediction->insert_rows(1,*trainingSampleYs[i]);
                if (prediction != nullptr) {
                    prediction->save("prediction_net" + ss.str() + ".dat", arma::raw_ascii);
                }
                this->outputQ(i);
                
            }
            this->outputPolicy();
            std::cout << "average duration " << experienceSet.size() / 1.0 / iter << std::endl;
        }
    }   
}

void NN_RLSolverMultiMLP::generateTrainingSample(){

    double maxQ;
    int action;
    std::shared_ptr<arma::mat> inputTemp(new arma::mat(netInputDim, 1));
    for (int i = 0; i < this->experienceSet.size(); i++) {
        this->getMaxQ(experienceSet[i].newState,&maxQ,&action);
//        std::cout << "maxQ:" <<maxQ << std::endl;
        double targetQ = experienceSet[i].reward +  trainingPara.discount()*maxQ;;
//        std::cout << "targetQ:" <<targetQ << std::endl;
        for ( int k = 0; k < this->stateDim; k++)
            inputTemp->at(k) =  experienceSet[i].oldState[k] / this->state_norm[k];
        int pos = trainingSampleXs[experienceSet[i].action]->n_cols;
        trainingSampleXs[experienceSet[i].action]->insert_cols(pos, *inputTemp);
        arma::mat target(1,1);
        target = targetQ;
        trainingSampleYs[experienceSet[i].action]->insert_cols(pos, target);
    }
}

void NN_RLSolverMultiMLP::outputQ(int m){


    arma::mat Q(n_rows, n_cols);
    std::shared_ptr<arma::mat> input = std::make_shared<arma::mat>(netInputDim, 1);
    for (int i = 0; i < n_rows; i++){
        for (int j = 0; j < n_cols; j++){
            input->at(0) = minx1 + i * dx1;
            input->at(1) = minx2 + j * dx2;
            input->at(0) /= this->state_norm[0];
            input->at(1) /= this->state_norm[1];
            
            nets[m]->setTrainingSamples(input, nullptr);
            nets[m]->forward();
            Q(i,j) = arma::as_scalar(*(nets[m]->netOutput()));
        
        }
    }
    std::stringstream ss;
    ss << m;
    Q.save("QMap" + ss.str() + ".dat", arma::raw_ascii);
}

void NN_RLSolverMultiMLP::outputPolicy(){
    arma::Mat<int> actionMap(n_rows, n_cols, arma::fill::zeros);
    std::shared_ptr<arma::mat> input = std::make_shared<arma::mat>(netInputDim, 1);
    arma::mat QMap(n_rows, n_cols, arma::fill::zeros);

    double maxQ;
    int action;
    for (int i = 0; i < n_rows; i++){
        for (int j = 0; j < n_cols; j++){
            maxQ = -1000;
            action = 0;
            for (int m = 0; m < numActions; m++){
                input->at(0) = minx1 + i * dx1;
                input->at(1) = minx2 + j * dx2;
                input->at(0) /= this->state_norm[0];
                input->at(1) /= this->state_norm[1];
            
                nets[m]->setTrainingSamples(input, nullptr);
                nets[m]->forward();
                double Qtemp = arma::as_scalar(*(nets[m]->netOutput()));
                if (Qtemp > maxQ) {
                    maxQ = Qtemp;
                    action = m;
                }
                
            }
            actionMap(i,j) = action;
            QMap(i,j) = maxQ;
        }
    }
    actionMap.save("actionMapNN.dat", arma::raw_ascii);
    QMap.save("QMaxMapNN.dat", arma::raw_ascii);
    

}