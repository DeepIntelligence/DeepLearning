#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include "Model_PoleFull.h"
using namespace ReinforcementLearning;
// this model is from paper THE POLE BALANCING PROBLEM A Benchmark Control Theory Problem
Model_PoleFull::Model_PoleFull(double dt0) {
    currState.resize(2);
    prevState.resize(2);
    dt = dt0;
    stateDim = 2;
    hiddenCurrState.resize(4);
    hiddenPrevState.resize(4);
    
    randNoise = std::make_shared<RandomStream>(-1, 1);
    numActions = 3;
}

void Model_PoleFull::run(int action) {
    double force;
    double accer_theta;
    double accer_x;
    switch (action) {
        case 0:
            force = -10 + randNoise->nextInt();
            break;
        case 1:
            force = 10 + randNoise->nextInt();
            break;
        case 2:
            force = randNoise->nextInt();
            break;
        default:break;
    }
    double l = 0.5;
    double massSum = 1.1;
    double massRatio = 1.0 / 11.0;
    double &theta = hiddenCurrState[0];
    double &theta_v = hiddenCurrState[1];
    double &x = hiddenCurrState[2];
    double &x_v = hiddenCurrState[3];
    
    accer_theta = 9.8 * sin(theta) - l * massRatio * 0.5 * pow(theta_v, 2.0) * sin(2.0 * theta) / 2.0 -   cos(theta) * force / massSum;
    accer_theta /= (4.0 * l / 3.0 - 0.1 * massRatio * l * cos(theta) * cos(theta));
    theta += theta_v * dt;
    if (theta > M_PI) theta -= 2.0*M_PI;
    if (theta < -M_PI) theta += 2.0*M_PI;
    theta_v += accer_theta * dt;
    
    accer_x = force / massSum + massRatio * l * (pow(theta_v, 2.0) * sin(theta) - accer_theta * cos(theta));
    x += x_v * dt;
    x_v += accer_x * dt;
    
    currState[0] = hiddenCurrState[0];
    currState[1] = hiddenCurrState[2];
}

void Model_PoleFull::createInitialState() {
    this->hiddenCurrState[0] = (randNoise->nextDou() - 0.5) * 0.0 * M_PI;
    this->hiddenCurrState[1] = 0.0;
    this->hiddenCurrState[2] = (randNoise->nextDou() - 0.5) * 0.0;
    this->hiddenCurrState[3] = 0.0;
    this->currState[0] = this->hiddenCurrState[0];
    this->currState[1] = this->hiddenCurrState[2];
}

