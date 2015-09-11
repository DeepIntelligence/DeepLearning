#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <algorithm>
#include "ProgramArgs.h"


using namespace std;


void ProgramArgs::LoadFromFile(const string & argsFilename) {
/** create an file object for reading file
    */ 
		ifstream argsFile (argsFilename);
		if (!argsFile.is_open()) {
			cerr << "Couldn't open " << argsFilename << endl;
			exit(1);
		}

		while (argsFile.good()) {
			string line;
			getline(argsFile, line);
			int beginComment = line.find("/*");
/** if /* is found
    */
			while (beginComment != string::npos) {
				string prefix = line.substr(0, beginComment);
				int endComment = line.find("*/");
				while (endComment == string::npos) {
					getline(argsFile, line);
					endComment = line.find("*/");
				}
				line = prefix + line.substr(endComment + 2);
				beginComment = line.find("/*");
			}

/** deal with // type of comment
    */
			beginComment = line.find("//");
			line = line.substr(0, beginComment); // still works if beginComment == npos


/** squeeze out the beginning and trailling space
    */
			int begin = 0;
			while (begin < line.size() && isspace(line[begin])) begin++;
			int end = line.size();
			while (end > 0 && isspace(line[end-1])) end--;
			line = line.substr(begin, end-begin);

			ParseArg(line);
		} 

		argsFile.close();
	}

void ProgramArgs::ParseArg(string argAndVal) {
		argAndVal.erase(remove_if(argAndVal.begin(), argAndVal.end(), ::isspace), argAndVal.end());

		if (argAndVal.size() == 0) return;

		cout << argAndVal << endl;

		int eq = argAndVal.find('=');
		if (eq == string::npos) {
			cerr << "Couldn't find value for line: " << argAndVal << endl;
			exit(1);
		}
		string arg = argAndVal.substr(0, eq);
		string val = argAndVal.substr(eq + 1);

                if (arg.compare("learningRate") == 0) learningRate = atof(val.c_str());
		else if (arg.compare("ntrain") == 0) ntrain = atoi(val.c_str());
		else if (arg.compare("ntest") == 0) ntest = atoi(val.c_str());
		else if (arg.compare("miniBatchSize") == 0) miniBatchSize = atoi(val.c_str());
		else if (arg.compare("momentum") == 0)  momentum= atof(val.c_str());
		else if (arg.compare("inputDim") == 0) inputDim = atoi(val.c_str());
		else if (arg.compare("hiddenDim") == 0) hiddenDim = atoi(val.c_str());
		else if (arg.compare("eps") == 0) eps = atof(val.c_str());
		else if (arg.compare("saveFrequency") == 0) saveFrequency = atoi(val.c_str());
		else if (arg.compare("dataPath") == 0) dataPath = val;
                else if (arg.compare("nEpoch") == 0) nEpoch = atoi(val.c_str());
                else if (arg.compare("learningRateDecay") == 0) learningRateDecay = atof(val.c_str());
                else if (arg.compare("dropOutFlag") == 0) dropOutFlag = atoi(val.c_str());
                else if (arg.compare("dropOutRate") == 0) dropOutRate = atof(val.c_str());
                else if (arg.compare("L2Decay") == 0) L2Decay = atof(val.c_str());
		else {
			cerr << "Unrecognized arg: " << arg << endl;
//			PrintHelp();
			exit(1);
		}
	}
/*
void ProgramArgs::CheckArgs() {
		bool training = (inModel.size() == 0);

		if (training) {
			for (int v = 0; v < 2; ++v) {
				if (numLayers[v] == -1) {
					cerr << "Number of layers not specified for view " << (v+1) << endl;
					exit(1);
				}

				if (numLayers[v] > 1 && hSize[v] == -1) {
					cerr << "Number of hidden units not specified for view " << (v+1) << endl;
					exit(1);
				}

				if (inData[v].size() == 0) {
					cerr << "Training data not specified for view " << (v + 1) << endl;
					exit(1);
				}
			}

			if (inParams.size() == 0) {
				cerr << "Training hyperparameter path not specified." << endl;
				exit(1);
			}

			if (outModel.size() == 0) {
				cerr << "Warning: model output path not specified" << endl;
			}

			if (outputSize == 0) {
				cerr << "Output size not specified" << endl;
				exit(1);
			}
		} else {
      // not training, read stored model
			if (outModel.size() > 0) {
				cerr << "Cannot specify both inModel and outModel" << endl;
				exit(1);
			}

      bool twoViews = (inData[0].size() > 0) && (inData[1].size() > 0);

			for (int v = 0; v < 2; ++v) {
        bool haveIn = (inData[v].size() > 0), haveOut = (outData[v].size() > 0);
				if (haveOut && !haveIn) {
					cerr << "Input data not specified for view " << (v + 1) << endl;
					exit(1);
				}

				if (haveIn && !haveOut) {
          if (!twoViews) {
            cerr << "Output data not specified for view " << (v + 1) << endl;
            exit(1);
          }
					cerr << "Warning: output data not specified for view " << (v + 1) << endl;
					cerr << "Computing correlation only." << endl;
				}
			}
		}

		for (int v = 0; v < 2; ++v) {
			if (inData[v].size() == 0 && iSize[v] == -1) {
				cerr << "Data size not specified for view " << (v + 1) << endl;
				exit(1);
			}
		}
	}

void ProgramArgs::PrintHelp() {
		printf("Arguments are specified on the command line with <name>=<value>. They can also\n");
		printf("   be read one per line from a file, allowing c-style comments.\n");
		printf("args         (string)   path to text file containing additional arguments\n");
		printf("inParams     (string)   path to binary file containing hyperparameter values\n");
		printf("outputSize   (int)      dimensionality of output representations\n");
		printf("numLayers1   (int)      number of layers in network for view 1\n");
		printf("numLayers2   (int)      number of layers in network for view 2\n");
		printf("hSize1       (int)      number of units per hidden layer for view 1\n");
		printf("hSize2       (int)      number of units per hidden layer for view 2\n");
		printf("inModel      (string)   path to previously stored DCCA model to read in\n");
		printf("outModel     (string)   path in which to store trained DCCA model\n");
		printf("inData1      (string)   path to matrix of input data for view 1\n");
		printf("inData2      (string)   path to matrix of input data for view 2\n");
		printf("outData1     (string)   path in which to store mapped data for view 1\n");
		printf("outData2     (string)   path in which to store mapped data for view 2\n");
		printf("iSize1       (int)      dimensionality of input data for view 1\n");
		printf("iSize2       (int)      dimensionality of input data for view 2\n");
		printf("iFeatSel1    (int)      reduced input dim of view 1 after PCA whitening\n");
		printf("iFeatSel2    (int)      reduced input dim of view 2 after PCA whitening\n");
		printf("trainSize    (int)      if specified, read only first n columns of all input\n");
	}
*/
ProgramArgs::ProgramArgs(int argc, char** argv)
{
		if (argc <= 1) {
                    cout << "enter parameters!" << endl;
			exit(0);
		}



    if (argc == 2) {
      LoadFromFile(argv[1]);
    } 
    
  cout << "parameter loading finish!";
		cout << endl;

}


