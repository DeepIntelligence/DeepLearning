#pragma once
#include <armadillo>
#include "model.h"
#include "../multilayerPerceptron/MultiLayerPerceptron.h"

namespace ReinforcementLearning{

struct RL_TrainingPara{
    RL_TrainingPara(){}
    int numEpisodes;
    int maxIter;
    int trainingSampleSize;
    double learningRate;
    double discount;
    State targetState;
    int defaultBigValue;
    bool experienceReplayFlag;
};

class NN_RLSolver {
public:
    NN_RLSolver(Model& m, NeuralNet::MultiLayerPerceptron& mlp0, RL_TrainingPara tp);
    void train();
    double getRewards(State oldS, State newS);
    double getCosts(const State& newS) const;
    bool targetReached(const State &s);
    void generatePolicy() const;
    void generateTrainingSample(std::shared_ptr<arma::mat> trainingSampleX, std::shared_ptr<arma::mat> trainingSampleY);
    

private:
    int inputDim, outputDim;
    Model& model;    
    NeuralNet::MultiLayerPerceptron& mlp;
    RL_TrainingPara trainingPara;
        std::shared_ptr<RandomStream> randChoice;



};

}
