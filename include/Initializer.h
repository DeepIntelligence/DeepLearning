#pragma once
#include <memory>
#include <armadillo>

namespace NeuralNet{
class Initializer{
public:
	virtual ~Initializer() {}
	virtual void applyInitialization(std::shared_ptr<arma::mat>) = 0;
};

class Initializer_normal: public Initializer{
public:
	Initializer_normal(double std0, double mean0):std(std0), mean(mean0){}
	virtual ~Initializer_normal(){}
	virtual void applyInitialization(std::shared_ptr<arma::mat> m){
		m->randn();
		m->transform([&](double val){ return val*std + mean;});
	}
private:
	double std, mean;

};

class Initializer_zero: public Initializer{
public:
	Initializer_zero(){}
	virtual ~Initializer_zero(){}
	virtual void applyInitialization(std::shared_ptr<arma::mat> m){
		m->zeros();
	}
};

class Initializer_identity: public Initializer{
public:
	Initializer_identity(){}
	virtual ~Initializer_identity(){}
	virtual void applyInitialization(std::shared_ptr<arma::mat> m){
		m->eye();
	}
};
class Initializer_glorot_normal: public Initializer{
public:
	Initializer_glorot_normal(){}
	virtual ~Initializer_glorot_normal(){}
	virtual void applyInitialization(std::shared_ptr<arma::mat> W){
		int inputDim = W->n_cols;
		int outputDim = W->n_rows;
		W->randu();
    	(*W) -= 0.5;
		(*W) *=sqrt(6.0/(inputDim+outputDim));
	}
};

class InitializerBuilder{
public:
	inline static std::shared_ptr<Initializer> GetInitializer(const DeepLearning::NeuralNetInitializerParameter para){
		switch (para.initializertype()) {
			case DeepLearning::NeuralNetInitializerParameter_InitializerType_normal:
			return std::shared_ptr<Initializer>(new Initializer_normal(para.normal_std(), para.normal_mean()));
			break;
			case DeepLearning::NeuralNetInitializerParameter_InitializerType_zero:
			return std::shared_ptr<Initializer>(new Initializer_zero);
			break;
			case DeepLearning::NeuralNetInitializerParameter_InitializerType_identity:
			return std::shared_ptr<Initializer>(new Initializer_identity);
			break;
			case DeepLearning::NeuralNetInitializerParameter_InitializerType_glorot_normal:
			return std::shared_ptr<Initializer>(new Initializer_glorot_normal);
			break;
			default:
			break;
		}
	}
};


}


