#include "MultiAddLayer.h"

using namespace NeuralNet;

void MultiAddLayer::activateUp(){
	*output = (*W_one) * (*inputOne) + (*W_two) * (*inputTwo);
}

void MultiAddLayer::initialize(){


}

