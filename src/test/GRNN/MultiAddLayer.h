#pragma once

namespace NeuralNet{

struct MultiAddLayer: public BaseLayer{






	virtual void activateUp();
	std::shared_ptr<arma::mat> inputOne, inputTwo;
	std::shared_ptr<arma::mat> W_one, W_two, B;
	std::shared_ptr<arma::mat> deltaOut_one, deltaOut_two;
	




};
}
