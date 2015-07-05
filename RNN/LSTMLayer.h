/* 
 * File:   LSTMLayer.h
 * Author: kai
 *
 * Created on July 1, 2015, 4:04 PM
 */

#ifndef LSTMLAYER_H
#define	LSTMLAYER_H

#include "../BaseLayer/BaseLayer.h"

namespace NeuralNet{

struct LSTMBrick: public BaseLayer{
		
	
	std::vector<LSTMBrick> inputSet;
	
}

}


class RNN_LSTM{


	

	forward();
	backward();
	
	
	std::vector<BaseLayer> inGateLayers, forgetGateLayers, cellStateLayers, outputGateLayers, informationLayers;
	std::vector<ElementWiseLayer>  outputLayers, forgetElementGateLayers, inputElementGateLayers;
	int numHiddenLayers;
	int inputDim, outputDim;

}


#endif	/* LSTMLAYER_H */

