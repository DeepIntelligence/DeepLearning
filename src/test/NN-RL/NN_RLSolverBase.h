#pragma once
#include <armadillo>
#include "BaseModel.h"
#include "common.h"
#include "Net.h"
#include "RLSolverBase.h"
#include "../Trainer/Trainer.h"

namespace ReinforcementLearning {

    class NN_RLSolverBase: public RLSolverBase {
    public:
        NN_RLSolverBase(std::shared_ptr<BaseModel> m,
                std::shared_ptr<NeuralNet::Net> net0,
                std::shared_ptr<NeuralNet::Trainer> trainer0, int Dim, DeepLearning::QLearningSolverParameter para);
        virtual ~NN_RLSolverBase(){}
        virtual void train() = 0;
        virtual void test(){}
        virtual void generateTrainingSample(std::shared_ptr<arma::mat> trainingSampleX, std::shared_ptr<arma::mat> trainingSampleY)=0;
        virtual void generateExperience() = 0;
	virtual double calQ(const State& S, int action) const = 0;
	virtual void getMaxQ(const State& S,double* Q, int* action);
        virtual double getRewards(const State& newS) const = 0;
        virtual bool terminate(const State& S) const = 0;
        virtual void setNormalizationConst() = 0;
    protected:
        int netInputDim;
        std::shared_ptr<NeuralNet::Net> net;
        std::shared_ptr<NeuralNet::Trainer> trainer;
        State state_norm;
        double action_norm;	

    };
}
