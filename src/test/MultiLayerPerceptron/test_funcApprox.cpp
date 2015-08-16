#include "MultiLayerPerceptron.h"
#include "common.h"
#include "../Trainer/Trainer.h"
using namespace NeuralNet;
using namespace DeepLearning;

void testComplex(char* filename);
void testSimple(char* filename);

int main(int argc, char** argv) {

    if (argc < 2) exit(1);
 
    testSimple(argv[1]);
      
    
    return 0;
}


void testSimple(char* filename){
    std::shared_ptr<arma::mat> X(new arma::mat(1,10));
    std::shared_ptr<arma::mat> Y(new arma::mat(1,10));
    
    for (int i = 0; i < X->n_elem; i++){
        X->at(i) = i;
    }
    
    double xmin = X->min();
    double xmax = X->max();
    X->transform([&](double x){return x/(xmax - xmin);});
    Y->ones();
    *Y = (*X); 
    Y->transform([](double val){return sin(val);});
    
    NeuralNetParameter nnpara;
    ReadProtoFromTextFile(filename, &nnpara);
//  nnpara.neuralnettrainingparameter().set_minibatchsize(X->n_elem);
    std::shared_ptr<Net> mlp(new MultiLayerPerceptron(nnpara));
    std::shared_ptr<Trainer> trainer(TrainerBuilder::GetTrainer(mlp, nnpara));
    
    mlp->setTrainingSamples(X,nullptr);
    mlp->forward();
    (mlp->netOutput())->print();
    trainer->setTrainingSamples(X, Y);
    trainer->train();
    Y->save("target.dat",arma::raw_ascii);
    mlp->netOutput()->save("trainingResult.dat",arma::raw_ascii);

}

void testComplex(char* filename){
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
    ReadProtoFromTextFile(filename, &nnpara);
//  nnpara.neuralnettrainingparameter().set_minibatchsize(X->n_elem);
    std::shared_ptr<Net> mlp(new MultiLayerPerceptron(nnpara));
    std::shared_ptr<Trainer> trainer(TrainerBuilder::GetTrainer(mlp, nnpara));
    
    mlp->setTrainingSamples(X,nullptr);
    mlp->forward();
    (mlp->netOutput())->print();
    trainer->setTrainingSamples(X, Y);
    trainer->train();
    Y->save("target.dat",arma::raw_ascii);
    mlp->netOutput()->save("trainingResult.dat",arma::raw_ascii);


}