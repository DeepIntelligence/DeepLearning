#include "Trainer.h"
#include "common.h"
#include "../MultiLayerPerceptron/MultiLayerPerceptron.h"

using namespace NeuralNet;
using namespace DeepLearning;
int main(int argc, char* argv[]){
    
    if (argc < 2) exit(1);
    
    NeuralNetParameter message; 
    ReadProtoFromTextFile(argv[1], &message);
    MultiLayerPerceptron mlp(message);
//    std::shared_ptr<Trainer> trainer(new)


    return 0;
}