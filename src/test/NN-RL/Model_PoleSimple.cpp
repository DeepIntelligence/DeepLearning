#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include "Model_PoleSimple.h"
using namespace ReinforcementLearning;
// this model is from paper Lease-squares policy iteration
Model_PoleSimple::Model_PoleSimple(double dt0) {
    currState.resize(2);
    prevState.resize(2);
    dt = dt0;
    stateDim = 2;
    randNoise = std::make_shared<RandomStream>(-10, 10);
    numActions = 3;
}

void Model_PoleSimple::run(int action) {
    double force;
    double accer;
    switch (action) {
        case 0:
            force = -50 + randNoise->nextInt();
            break;
        case 1:
            force = 50 + randNoise->nextInt();
            break;
        case 2:
            force = randNoise->nextInt();
            break;
        default:break;
    }
    double &theta = currState[0];
    double &theta_v = currState[1];
    accer = 9.8 * sin(theta) - 0.1 * 2.0 * 0.5 * pow(theta_v, 2.0) * sin(2.0 * theta) / 2.0 - 0.1 * cos(theta) * force;
    accer /= (4.0 * 0.5 / 3.0 - 0.1 * 2.0 * 0.5 * cos(theta) * cos(theta) );
    prevState = currState;
    theta += theta_v * dt;
    if (theta > M_PI) theta -= 2.0*M_PI;
    if (theta < -M_PI) theta += 2.0*M_PI;
    theta_v += accer * dt;
}

void Model_PoleSimple::createInitialState() {
    this->currState[0] = (randNoise->nextDou() - 0.5) * 0.1 * M_PI;
    this->currState[1] = 0.0;
}

