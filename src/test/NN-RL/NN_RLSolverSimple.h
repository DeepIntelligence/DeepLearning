#pragma once
#include <armadillo>
#include "BaseModel.h"
#include "Model_PoleSimple.h"
#include "NN_RLSolverBase.h"
#include "Net.h"
#include "../Trainer/Trainer.h"

namespace ReinforcementLearning {
    class NN_RLSolverSimple: public NN_RLSolverBase {
    public:
        NN_RLSolverSimple(std::shared_ptr<BaseModel> m,
                std::shared_ptr<NeuralNet::Net> net0,
                std::shared_ptr<NeuralNet::Trainer> trainer0,
                int Dim, DeepLearning::QLearningSolverParameter para);
        virtual ~NN_RLSolverSimple(){}
        virtual double getRewards(const State& newS) const;
        virtual bool terminate(const State& S) const;
        virtual void setNormalizationConst();
        virtual void test(){}
    private:
        std::vector<double> durationVec;
    };
}
