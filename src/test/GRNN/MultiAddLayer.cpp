#include "MultiAddLayer"


using namespace NeuralNet;

virtual void activateUp(){
	*output = (*W_one) * (*inputOne) + (*W_two) * (*inputTwo);



}
