#pragma once
#include <armadillo>
#include "BaseModel.h"
#include "common.h"
#include "Model_PoleSimple.h"
#include "Net.h"
#include "../Trainer/Trainer.h"

namespace ReinforcementLearning {

    class NN_RLSolverBase {
    public:
        NN_RLSolverBase(std::shared_ptr<BaseModel> m,
                std::shared_ptr<NeuralNet::Net> net0,
                std::shared_ptr<NeuralNet::Trainer> trainer0, int Dim, DeepLearning::QLearningSolverParameter para);
        virtual ~NN_RLSolverBase(){}
        virtual void train();
        virtual void test(){}
        virtual void generateTrainingSample(std::shared_ptr<arma::mat> trainingSampleX, std::shared_ptr<arma::mat> trainingSampleY);
        virtual void generateExperience();
        virtual void getMaxQ(const State& S,double* Q, int* action);
        virtual double getRewards(const State& newS) const;
        virtual bool terminate(const State& S) const;
        virtual void setNormalizationConst(){}
    protected:
        int stateDim;
        int netInputDim;
        std::shared_ptr<BaseModel> model;
        std::shared_ptr<NeuralNet::Net> net;
        std::shared_ptr<NeuralNet::Trainer> trainer;
        DeepLearning::QLearningSolverParameter trainingPara;
        std::shared_ptr<RandomStream> randChoice;
        std::vector<Experience> experienceSet;
        State state_norm;
        double action_norm;
    };
}
