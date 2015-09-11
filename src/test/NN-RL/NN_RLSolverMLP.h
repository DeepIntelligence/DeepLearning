#pragma once
#include <armadillo>
#include "BaseModel.h"
#include "Model_PoleSimple.h"
#include "NN_RLSolverBase.h"
#include "Net.h"
#include "../Trainer/Trainer.h"

namespace ReinforcementLearning {
    class NN_RLSolverMLP: public NN_RLSolverBase {
    public:
        NN_RLSolverMLP(std::shared_ptr<BaseModel> m,
                std::shared_ptr<NeuralNet::Net> net0,
                std::shared_ptr<NeuralNet::Trainer> trainer0,
                int Dim, DeepLearning::QLearningSolverParameter para);
        virtual ~NN_RLSolverMLP(){}
        virtual void train();
        virtual void generateTrainingSample(std::shared_ptr<arma::mat> trainingX, std::shared_ptr<arma::mat> trainingY);
	virtual void generateExperience();
	virtual double getRewards(const State& newS) const;
        virtual bool terminate(const State& S) const;
        virtual void setNormalizationConst();
        virtual double calQ(const State& S, int action) const;
        virtual void test(){}
    protected:
        std::vector<Experience> experienceSet;        
    private:
        std::vector<double> durationVec;
    };
}
