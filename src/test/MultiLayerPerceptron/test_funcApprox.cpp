#include "MultiLayerPerceptron.h"
#include "common.h"

using namespace NeuralNet;
using namespace DeepLearning;

int main(int argc, char** argv) {

    if (argc < 2) exit(1);
    std::shared_ptr<arma::mat> X(new arma::mat(1,100));
    std::shared_ptr<arma::mat> Y(new arma::mat(1,100));
    
    for (int i = 0; i < X->n_elem; i++){
        X->at(i) = i;
    }
    
    double xmin = X->min();
    double xmax = X->max();
    X->transform([&](double x){return x/(xmax - xmin);});
    Y->ones();
    *Y = 5*(*X); 
    Y->transform([](double val){return sin(4*val) + exp(val);});
    
    NeuralNetParameter nnpara;
    ReadProtoFromTextFile(argv[1], &nnpara);
//  nnpara.neuralnettrainingparameter().set_minibatchsize(X->n_elem);
    MultiLayerPerceptron mlp(nnpara);
    mlp.setTrainingSample(X, Y);
    mlp.train();
    Y->save("target.dat",arma::raw_ascii);
    mlp.getNetOutput()->save("trainingResult.dat",arma::raw_ascii);
    
    //MLPTrainer mlpTrainer(mlp);
    //Optimization::LBFGS::LBFGS_param param(100,20, 50 , "result.txt");
    //Optimization::LBFGS lbfgs_opt(mlpTrainer,param, Optimization::LBFGS::Wolfe);
    //lbfgs_opt.minimize();
    
    
    
    
    
    
    return 0;
}

