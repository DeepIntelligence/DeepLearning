#pragma once
#include <armadillo>
#include "BaseModel.h"
#include "Model_PoleSimple.h"
#include "NN_RLSolverMLP.h"
#include "Net.h"
#include "../Trainer/Trainer.h"

namespace ReinforcementLearning {
    class NN_RLSolverRNN: public NN_RLSolverMLP {
    public:
        NN_RLSolverRNN(std::shared_ptr<BaseModel> m,
                std::shared_ptr<NeuralNet::Net> net0,
                std::shared_ptr<NeuralNet::Trainer> trainer0,
                int Dim, DeepLearning::QLearningSolverParameter para);
        virtual ~NN_RLSolverRNN(){}
        virtual void train();
        virtual void generateExperience();
        virtual void generateTrainingSampleVec(std::vector<std::shared_ptr<arma::mat>>& trainingSampleX, 
        std::vector<std::shared_ptr<arma::mat>>& trainingSampleY);
        virtual bool terminate(const State& S) const;
        virtual void setNormalizationConst();
        virtual double calQ(const State& S, int action) const;
        virtual void test();
        void outputTraining(std::vector<std::shared_ptr<arma::mat>> &trainingXVec,std::string);
    protected:
        std::vector<double> durationVec;
        std::vector<std::shared_ptr<arma::mat>> trainingXVec, trainingYVec;
        std::vector<std::vector<Experience>> experienceSeqVec;
    };
}
