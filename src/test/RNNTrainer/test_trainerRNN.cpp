#include "Trainer.h"
#include "common.h"
#include "../MultiLayerPerceptron/MultiLayerPerceptron.h"

using namespace NeuralNet;
using namespace DeepLearning;
int main(int argc, char* argv[]){
    
    if (argc < 2) exit(1);
    
    NeuralNetParameter message; 
    ReadProtoFromTextFile(argv[1], &message);

    std::shared_ptr<arma::mat> DataX(new arma::mat);
    std::shared_ptr<arma::mat> DataY(new arma::mat);
    std::shared_ptr<arma::mat> trainDataX(new arma::mat);
    std::shared_ptr<arma::mat> trainDataY(new arma::mat);
    std::shared_ptr<arma::mat> testDataX(new arma::mat);
    std::shared_ptr<arma::mat> testDataY(new arma::mat);
    std::shared_ptr<arma::mat> ValidationDataX(new arma::mat);
    std::shared_ptr<arma::mat> ValidationDataY(new arma::mat);

    loadData_MNIST(DataX,DataY,(std::string)argv[2]);

    int ntrain =2000;
    int ntest = 1000;
//  now I split data into train, test, and validation
    trainDataX = std::make_shared<arma::mat>(DataX->cols(0,ntrain-1));
    trainDataY = std::make_shared<arma::mat>(DataY->cols(0,ntrain-1));
    testDataX = std::make_shared<arma::mat>(DataX->cols(ntrain,ntrain+ntest-1));
    testDataY = std::make_shared<arma::mat>(DataY->cols(ntrain,ntrain+ntest-1));
    
        
    std::shared_ptr<Net> mlp(new MultiLayerPerceptron(message));
    std::shared_ptr<Trainer> trainer( TrainerBuilder::GetTrainer(mlp,message));
    trainer->setTrainingSamples(trainDataX, trainDataY);
    trainer->train();
    return 0;
}