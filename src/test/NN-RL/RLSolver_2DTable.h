#pragma once
#include <utility>
#include "common.h"
#include "RLSolverBase.h"


namespace ReinforcementLearning {

    class RLSolver_2DTable : RLSolverBase{
        public:
        RLSolver_2DTable(std::shared_ptr<BaseModel> m, int Dim, 
        DeepLearning::QLearningSolverParameter para, int n_row0, int n_col0, 
        double dx, double dy, double min_x, double min_y);

        virtual ~RLSolver_2DTable() {}
        virtual void train();
        virtual void test();
        void replayExperience();
        virtual void updateQ(Experience);
        virtual void getMaxQ(const State& S, double* Q, int* action) const;
        arma::cube& getQTable(){return QTable;}
        private:
        void outputPolicy();
	void outputQ(std::string filename);
        void writeTrajectory(int iter, std::ostream &os, int action, State state, double reward) const;
        std::pair<int, int> stateToIndex(const State & S) const;
        arma::cube QTable;
        int n_rows, n_cols, numActions;
        double dx1, dx2, minx1, minx2;
        arma::Mat<int> count;
        std::vector<Experience> experienceVec;
    };
}
