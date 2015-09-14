#pragma once
#include <array>
namespace ReinforcementLearning {

    typedef std::vector<double> State;

    class BaseModel {
    public:
        virtual ~BaseModel(){}
        virtual void run(int action) = 0;
	virtual void run(int action, int steps){
		for (int i = 0; i < steps; i++){
			run(action);
		}
	};
        virtual State getCurrState() {
            return currState;
        }
        virtual void createInitialState() = 0;
        virtual int getNumActions(){ return numActions;}
        virtual double getRewards() const{}
        virtual bool terminate() const{}
    protected:
        State currState, prevState;
        int numActions;
        int stateDim;
    };
    
    struct Experience{
        State oldState, newState;
        int action;
        double reward;
        Experience(State old0, State new0, int a0, double c0):
        oldState(old0),newState(new0), action(a0), reward(c0)
        {}
    };
    
}
