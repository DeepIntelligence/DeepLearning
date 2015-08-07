#include "MultiLayerPerceptron.h"
#include "common.h"
#include "../Trainer/Trainer.h"
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
    Y->transform([](double val){return sin(4*val);});
    
    NeuralNetParameter nnpara;
    ReadProtoFromTextFile(argv[1], &nnpara);
//  nnpara.neuralnettrainingparameter().set_minibatchsize(X->n_elem);
    std::shared_ptr<Net> mlp(new MultiLayerPerceptron(nnpara));
    std::shared_ptr<Trainer> trainer(TrainerBuilder::GetTrainer(mlp, nnpara));
    
    trainer->setTrainingSamples(X, Y);
    trainer->train();
    Y->save("target.dat",arma::raw_ascii);
    mlp->netOutput()->save("trainingResult.dat",arma::raw_ascii);
    
    //MLPTrainer mlpTrainer(mlp);
    //Optimization::LBFGS::LBFGS_param param(100,20, 50 , "result.txt");
    //Optimization::LBFGS lbfgs_opt(mlpTrainer,param, Optimization::LBFGS::Wolfe);
    //lbfgs_opt.minimize();
    
    
    
    
    
    
    return 0;
}

