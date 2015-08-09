#pragma once
#include <armadillo>
#include "BaseModel.h"
#include "Model_PoleSimple.h"
#include "Net.h"
#include "../Trainer/Trainer.h"

namespace ReinforcementLearning {

    struct RL_TrainingPara {

        RL_TrainingPara() {}
        int numEpisodes;
        int maxIter;
        int trainingSampleSize;
        double learningRate;
        double discount;
        int defaultBigValue;
        bool experienceReplayFlag;
    };
    class NN_RLSolver {
    public:
        NN_RLSolver(std::shared_ptr<BaseModel> m,
                std::shared_ptr<NeuralNet::Net> net0,
                std::shared_ptr<NeuralNet::Trainer> trainer0,
                RL_TrainingPara tp, int Dim);
        virtual ~NN_RLSolver(){}
        virtual void train();
        virtual double getRewards(const State& newS) const;
        void generatePolicy() const;
        virtual void generateTrainingSample(std::shared_ptr<arma::mat> trainingSampleX, std::shared_ptr<arma::mat> trainingSampleY);
        virtual void generateExperience();
        virtual bool terminate(const State& S) const;
        virtual void setNormalizationConst();
        virtual void getMaxQ(const State& S,double* Q, int* action);
 
    private:
        int stateDim;
        int netInputDim, outputDim;
        std::shared_ptr<BaseModel> model;
        std::shared_ptr<NeuralNet::Net> net;
        std::shared_ptr<NeuralNet::Trainer> trainer;
        RL_TrainingPara trainingPara;
        std::shared_ptr<RandomStream> randChoice;
        std::vector<Experience> experienceSet;
        State state_norm;
        double action_norm;
        std::vector<double> durationVec;
    };

}
