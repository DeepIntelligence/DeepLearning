#pragma once


namespace NeuralNet{

class Layer{
public:
	virtual ~Layer(){}
	virtual void activateUp() = 0;


};
}
