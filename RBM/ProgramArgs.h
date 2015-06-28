#pragma once

#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>

using namespace std;

struct ProgramArgs {
    ProgramArgs(int argc, char *argv[]);
	 void LoadFromFile(const string & argsFilename);
	 void ParseArg(string argAndVal); 
	 int ntrain, ntest, saveFrequency, inputDim, hiddenDim,nEpoch;
         double learningRate, eps, momentum, miniBatchSize, learningRateDecay; 
         string dataPath;
};
