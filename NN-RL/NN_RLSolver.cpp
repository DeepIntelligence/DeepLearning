#include "NN_RLSolver.h"

using namespace ReinforcementLearning;
NN_RLSolver::NN_RLSolver(Model& m, NeuralNet::MultiLayerPerceptron& mlp0, RL_TrainingPara tp):
    model(m), mlp(mlp0), trainingPara(tp){
    randChoice = std::make_shared<RandomStream>(-10, 10);
}

void NN_RLSolver::train(){
    std::shared_ptr<arma::mat> trainingSampleX, trainingSampleY;
    std::shared_ptr<arma::mat> trainingSampleX_total, trainingSampleY_total;   
    for (int iter = 0; iter < trainingPara.maxIter; iter++){
        this->generateTrainingSample(trainingSampleX, trainingSampleY);
        if (trainingPara.experienceReplayFlag) {
            trainingSampleX_total->insert_cols(trainingSampleX_total->n_cols-1,*trainingSampleX);
            trainingSampleY_total->insert_cols(trainingSampleY_total->n_cols-1,*trainingSampleY);
        } else {
            trainingSampleX_total = trainingSampleX;
            trainingSampleY_total = trainingSampleY;
        }
        
        mlp.setTrainingSample(trainingSampleX_total, trainingSampleY_total);
        mlp.train();
        
    }
    
    mlp.save("result.txt");
    this->generatePolicy();
    
}

void NN_RLSolver::generateTrainingSample(std::shared_ptr<arma::mat> trainingSampleX, std::shared_ptr<arma::mat> trainingSampleY){
    trainingSampleX = std::make_shared<arma::mat>(inputDim,1);
    trainingSampleY = std::make_shared<arma::mat>(1,1);
    double minQ;
    arma::mat outputTemp(1,1);
    std::shared_ptr<arma::mat> inputTemp(new arma::mat);
    
    for( int i = 0; i < trainingPara.trainingSampleSize; i++){
                
        int action = randChoice->nextInt();
        State currState = model.getCurrState();
        model.run(action);
        inputTemp->at(0) = currState.theta;
        inputTemp->at(1) = currState.theta_v; 
        inputTemp->at(2) = action;
        mlp.feedForward(inputTemp);
        for ( int j = 0 ; j < 3; j++){
            inputTemp->at(0) = currState.theta;
            inputTemp->at(1) = currState.theta_v; 
            inputTemp->at(2) = j; 
            double tempQ = arma::as_scalar(*(mlp.getNetOutput()));
            if( minQ > tempQ ) minQ = tempQ;
        }
        double targetQ = this->getCosts(currState) + minQ;;
        outputTemp(1) = targetQ;
        if ( i == 0 ){
            *trainingSampleX = *inputTemp;
            trainingSampleY->at(0) = targetQ;
        } else {
            trainingSampleX->insert_cols(i,*inputTemp);
            trainingSampleY->insert_cols(i,outputTemp);
        }
        if( model.terminate() ) break;
    }
}

void NN_RLSolver::generatePolicy() const{

}

double NN_RLSolver::getCosts(const State &newS) const{
    return 0.0;
}

