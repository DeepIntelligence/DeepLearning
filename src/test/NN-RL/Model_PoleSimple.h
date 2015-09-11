/*
This model is the Inverted Pendulum problem found in the paper
"lease-squared policy iterations"
*/
#pragma once
#include <string>
#include <cmath>
#include <iostream>
#include "Util.h"
#include "BaseModel.h"

using namespace NeuralNet;
namespace ReinforcementLearning {  
class Model_PoleSimple: public BaseModel{
public:
    Model_PoleSimple(double dt0);
    ~Model_PoleSimple(){}
    virtual void run(int action);
    virtual void createInitialState();
private:
    std::shared_ptr<RandomStream> randNoise;    
    double dt;
};
}