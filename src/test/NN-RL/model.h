/*
This model is the Inverted Pendulum problem found in the paper
"lease-squared policy iterations"

*/


#pragma once
#include <string>
#include <cmath>
#include <iostream>
#include "Util.h"

using namespace NeuralNet;
struct State{
	double theta;
	double theta_v;
	friend std::ostream& operator<<(std::ostream& out, State& in){
		std::cout << in.theta <<"  " << in.theta_v;
                return std::cout;
	}
};

class Model{
public:
    Model(double dt0);
    void run(int action);
    double getCosts(State) const;
    State getCurrState() const{return currState;};
    void createInitialState();
    bool terminate(){ return (currState.theta < -M_PI || currState.theta > M_PI);}
private:
    State oldState, currState;
    std::shared_ptr<RandomStream> randNoise;    
    double dt;

};
