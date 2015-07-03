

namespace NeuralNet{

struct ElementwiseLayer{
	
	ElementwiseLayer(int inputSize0){
            inputSize = inputSize0;
		inputOne = std::make_shared<arma::mat>();
		inputTwo = std::make_shared<arma::mat>();
		output = std::make_shared<arma::mat>();
        };

	int inputSize;
	
	std::shared_ptr<arma::mat> inputOne, inputTwo, output;

 	void activateUp();

};

}
