

namespace NeuralNet{

struct ElementwiseLayer{
	
	ElementwiseLayer(int inputSize0){
        inputSize = inputSize0;
		//we only need to assign memory to the output
		delta_outOne = std::make_shared<arma::mat>();
		delta_outTwo = std::make_shared<arma::mat>(); 
		output = std::make_shared<arma::mat>();
		};

	
	std::shared_ptr<arma::mat> inputOne, inputTwo, output;
	std::shared_ptr<arma::mat> delta_outOne, delta_outTwo;
 	void activateUp();
	void updatePara(std::shared_ptr<arma::mat> delta_in);
	
	int inputSize;
};

}
