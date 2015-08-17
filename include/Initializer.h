#pragma once
#include <memory>
#include <armadillo>

namespace NeuralNet{
class Initializer{
public:
	virtual ~Initializer() {}
	virtual void applyInitialization(std::shared_ptr<arma::mat>) = 0;
};

class InitializerBuilder{
public:
	inline static std::shared_ptr<Initializer> GetInitializer(const DeepLearning::NeuralNetInitializerParameter_InitializerType type){
		switch (type) {
			case DeepLearning::NeuralNetInitializerParameter_InitializerType_normal:
			break;
			case DeepLearning::NeuralNetInitializerParameter_InitializerType_zero:
			break;
			default:
			break;
		}
	}
};

class Initializer_normal{
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

class Initializer_zero{
public:
	Initializer_zero(){}
	virtual ~Initializer_zero(){}
	virtual void applyInitialization(std::shared_ptr<arma::mat> m){
		m->zeros();
	}
};

class Initializer_identity{
public:
	Initializer_identity(){}
	virtual ~Initializer_identity(){}
	virtual void applyInitialization(std::shared_ptr<arma::mat> m){
		m->eye();
	}
};

}


