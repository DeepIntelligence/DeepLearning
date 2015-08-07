#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include "model.h"
// this model is from paper Lease-squares policy iteration
Model::Model(double dt0) {
    dt = dt0;
    randNoise = std::make_shared<RandomStream>(-10, 10);
}

void Model::run(int action) {
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
    accer = 9.8 * sin(currState.theta) - 0.1 * 2.0 * 0.5 * pow(currState.theta_v, 2.0) * sin(2.0 * currState.theta) / 2.0 - 0.1 * cos(currState.theta) * force;
    accer /= (4.0 * 0.5 / 3.0 - 0.1 * 2.0 * 0.5 * cos(currState.theta));
    oldState = currState;
    currState.theta += currState.theta_v * dt;
    if (currState.theta > M_PI) currState.theta -= M_PI - 
    currState.theta_v += accer * dt;
}

void Model::createInitialState() {
    currState.theta = (randNoise->nextDou() - 0.5) * M_PI;
    currState.theta_v = 0.0;
}

