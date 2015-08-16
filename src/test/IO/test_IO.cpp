#include <iostream>
#include "common.h"
using namespace DeepLearning;
int main(int argc, char *argv[]){

	NeuralNetParameter message;
	RNNStructParameter submessage;
	if (argc == 2){

		ReadProtoFromTextFile(argv[1], &message);
	std::cout << message.layerstruct_size() << std::endl;
	for (int i = 0 ; i < message.layerstruct_size(); i++ ){
		if (message.layerstruct(i).has_name()) 
			std::cout << message.layerstruct(i).name() << std::endl;	
		if (message.layerstruct(i).has_activationtype()){
			std::cout << message.layerstruct(i).activationtype() << std::endl;
			if( message.layerstruct(i).activationtype() == LayerStructParameter_ActivationType_sigmoid)
				std::cout << "good" << std::endl;	
		}
	}

	std::cout << message.neuralnettrainingparameter().learningrate() << std::endl;
	std::cout << message.neuralnettrainingparameter().minibatchsize()<< std::endl;
	std::cout << message.neuralnettrainingparameter().nepoch() << std::endl;
	std::cout << message.neuralnettrainingparameter().epi() << std::endl;
	std::cout << message.neuralnettrainingparameter().trainertype() << std::endl;

	std::cout << std::endl;
	
	std::cout << "test Kai message" << std::endl;
	std::cout << message.rnnstruct().numhiddenlayers() << std::endl;
	std::cout << message.rnnstruct().hiddenlayeroutputdim() << std::endl;

	submessage = message.rnnstruct();

	std::cout << "test sub message" << std::endl;
	std::cout << submessage.numhiddenlayers() << std::endl;
	std::cout << submessage.hiddenlayeroutputdim() << std::endl;	


	}

	return 0;
}
