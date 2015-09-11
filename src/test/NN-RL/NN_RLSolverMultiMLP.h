#pragma once
#include <armadillo>
#include "BaseModel.h"
#include "Model_PoleSimple.h"
#include "NN_RLSolverBase.h"
#include "NN_RLSolverMLP.h"
#include "Net.h"
#include "../Trainer/Trainer.h"

namespace ReinforcementLearning {
    class NN_RLSolverMultiMLP: public NN_RLSolverMLP {
    public:
        NN_RLSolverMultiMLP(std::shared_ptr<BaseModel> m,
                std::vector<std::shared_ptr<Net>> net0,
                std::shared_ptr<NeuralNet::Trainer> trainer0,
                int Dim, DeepLearning::QLearningSolverParameter para);
        virtual ~NN_RLSolverMultiMLP(){}
        virtual void train();
        virtual void generateTrainingSample();
        virtual double calQ(const State& S, int action) const;
        void outputPolicy();        
    private:
	void outputQ(int i);
        int numActions;
    	std::vector<std::shared_ptr<NeuralNet::Net>> nets;
    	std::vector<std::shared_ptr<arma::mat>> trainingSampleXs, trainingSampleYs;
        std::vector<double> durationVec;
        
    // parameters to output the Q value, for checking purpose
        int n_rows;
        int n_cols;
        double dx1;
        double dx2;
        double minx1;
        double minx2;

        
    };
}
