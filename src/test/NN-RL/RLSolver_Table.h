#pragma once
#include <utility>
#include "RLSolverBase.h"


namespace ReinforcementLearning {

    class RLSolver_Table : RLSolverBase{
        public:
        RLSolver_Table(std::shared_ptr<BaseModel> m, int Dim, DeepLearning::QLearningSolverParameter para);

        virtual ~RLSolver_Table() {}
        virtual void train();
        virtual double getRewards(const State & newS) const;
        virtual bool terminate(const State & S) const;
        virtual void updateQ(Experience);
        virtual void getMaxQ(const State& S, double* Q, int* action) const;
        private:
        void outputPolicy();
	void outputQ(std::string filename);
        std::pair<int, int> stateToIndex(const State & S) const;
        arma::cube QTable;
        int n_rows, n_cols, numActions;
        double dx1, dx2, minx1, minx2;
        arma::Mat<int> count;
    };
}
