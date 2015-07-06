#include <memory>
#include <armadillo>
/*
  generate simulation data for lstm
  x(t) = x(t-1) + 0.3x(t-3)
  y(t) = sigmoid(x(t)) 
*/

class SimGenerate{
	
public:
	
	void SimGenerate();
	void initialize();
	void generateSim();
	
	int seqLength;
	int sampleNum;
	int featureDim;

	std::shared_ptr<arma::mat> X;
	std::shared_ptr<arma::mat> Y;

}


SimGenerate::SimGenerate(int seqLength0, int sampleNum0, int featureDim0){
	seqLength = seqLength0;
	sampleNum = sampleNum0;
	featureDim = featureDim0;
	
	initialize();
	generateSim();
}

void SimGenerate::initialize(){
	for(int i=0;i<featureDim;i++){
		(*X)[0][i] = randu();
		(*X)[1][i] = randu();
		(*X)[2][i] = randu();
		
		(*Y)[0][i] = (*X)[0][i];
		(*Y)[0][i].transform([&](double val){return 1/(1+exp(-val));});
		(*Y)[1][i] = (*X)[1][i];
		(*Y)[1][i].transform([&](double val){return 1/(1+exp(-val));});
		(*Y)[2][i] = (*X)[2][i];
		(*Y)[2][i].transform([&](double val){return 1/(1+exp(-val));});
	}
}

void SimGenerate::generateSim(){
	for (int t=3;t<X.row();t++){
		(*X)[t] = (*X)[t-1] + 0.3*((*X)[t-3]);
		(*Y)[t] = (*X)[t];
		(*Y)[t].transform([&](double val){return 1/(1+exp(-val));});
	}
}