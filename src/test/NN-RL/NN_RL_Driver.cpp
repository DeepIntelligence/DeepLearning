#include "Model_PoleSimple.h"
#include "Model_PoleFull.h"
#include "BaseModel.h"
#include "NN_RLSolverBase.h"
#include "NN_RLSolverMLP.h"
#include "NN_RLSolverMultiMLP.h"
#include "NN_RLSolverRNN.h"
#include "MultiLayerPerceptron.h"
#include "RNN.h"
#include "ElmanRL.h"
#include "Util.h"
#include "common.h"
#include "RLSolver_2DTable.h"
void testModel();
void testModelFull();
void testSolverMLP(char* filename, char*);
void testSolverMultiMLP(char* filename, char*);
void testSolverRNN(char* filename, char*);
void testSolverElman(char* filename, char*);
void testTableSolver(char* filename);
using namespace ReinforcementLearning;
using namespace DeepLearning;
using namespace NeuralNet;
int main(int argc, char* argv[]){

 //   for (int i = 0; i < 100; i++)
 //       testModel();
 //   testModelFull();
 	testTableSolver(argv[1]);
//    testSolverMultiMLP(argv[1], argv[2]);
 //   testSolverRNN(argv[1],argv[2]);
//    testSolverElman(argv[1],argv[2]);
    return 0;
}
#if 1
void testSolverMultiMLP(char* filename1, char* filename2){
//without learning, the average balancing duration is below 12
    double dt = 0.1;
    NeuralNetParameter message1;
    ReinforcementLearningParameter message2;
    QLearningSolverParameter message3;
    ReadProtoFromTextFile(filename1, &message1);
    ReadProtoFromTextFile(filename2, &message2);
    message3 = message2.qlearningsolverparameter();
    arma::arma_rng::set_seed_random();
    std::shared_ptr<BaseModel> model(new Model_PoleSimple(dt));    
    
    std::vector<std::shared_ptr<Net>> net;
    for (int i = 0; i < model->getNumActions(); i++)
        net.push_back(std::shared_ptr<Net>(new MultiLayerPerceptron(message1)));
    std::shared_ptr<Trainer> trainer(TrainerBuilder::GetTrainer(net[0],message1));
    NN_RLSolverMultiMLP rlSolver(model, net, trainer, 2, message3);
    rlSolver.train();
}



void testSolverMLP(char* filename1, char* filename2){
//without learning, the average balancing duration is below 12
    double dt = 0.1;
    NeuralNetParameter message1;
    ReinforcementLearningParameter message2;
    QLearningSolverParameter message3;
    ReadProtoFromTextFile(filename1, &message1);
    ReadProtoFromTextFile(filename2, &message2);
    message3 = message2.qlearningsolverparameter();
    arma::arma_rng::set_seed(1);
    std::shared_ptr<Net> net(new MultiLayerPerceptron(message1));
    std::shared_ptr<Trainer> trainer(TrainerBuilder::GetTrainer(net,message1));
    std::shared_ptr<BaseModel> model(new Model_PoleSimple(dt));
    NN_RLSolverMLP rlSolver(model, net, trainer, 2, message3);
    rlSolver.train();
    net->save("trainedresult");
}

void testTableSolver(char* filename2){
    double dt = 0.1;
    ReinforcementLearningParameter message2;
    QLearningSolverParameter message3;
    ReadProtoFromTextFile(filename2, &message2);
    message3 = message2.qlearningsolverparameter();
    arma::arma_rng::set_seed(1);
    std::shared_ptr<BaseModel> model(new Model_PoleSimple(dt));
    
    int n_rows = 40;
    int n_cols = 30;
    
    double dx1 = 2.0*M_PI / n_rows;
    double dx2 = 30.0 / n_cols;
    double minx1 = -1.0*M_PI;
    double minx2 = -15.0;
    RLSolver_2DTable rlSolver(model, 2, message3, n_rows, n_cols, dx1, dx2, minx1, minx2);
    rlSolver.train();
    rlSolver.test();
    
}

void testSolverRNN(char* filename1, char* filename2){

    double dt = 0.05;
    NeuralNetParameter message1;
    ReinforcementLearningParameter message2;
    QLearningSolverParameter message3;
    ReadProtoFromTextFile(filename1, &message1);
    ReadProtoFromTextFile(filename2, &message2);
    message3 = message2.qlearningsolverparameter();
    arma::arma_rng::set_seed(1);
    std::shared_ptr<Net> net(new RNN(message1));
    std::shared_ptr<Trainer> trainer(TrainerBuilder::GetTrainer(net,message1));
    std::shared_ptr<BaseModel> model(new Model_PoleFull(dt));
    NN_RLSolverRNN rlSolver(model, net, trainer, 2, message3);
    rlSolver.train();
    net->save("trainedresult");
}


void testSolverElman(char* filename1, char* filename2){

    double dt = 0.05;
    NeuralNetParameter message1;
    ReinforcementLearningParameter message2;
    QLearningSolverParameter message3;
    ReadProtoFromTextFile(filename1, &message1);
    ReadProtoFromTextFile(filename2, &message2);
    message3 = message2.qlearningsolverparameter();
    arma::arma_rng::set_seed(1);
    std::shared_ptr<Net> net(new ElmanRL(message1));
    std::shared_ptr<Trainer> trainer(TrainerBuilder::GetTrainer(net,message1));
    std::shared_ptr<BaseModel> model(new Model_PoleFull(dt));
    NN_RLSolverRNN rlSolver(model, net, trainer, 2, message3);
    rlSolver.train();
//    net->save("trainedresult");
}


void testModel(){
// dt = 0.1 is from the paper Neural Fitted Q iteration - First Experiences with..   
    double dt = 0.1;
    Model_PoleSimple model(dt);
    
    model.createInitialState();
    State state(2);
    double T = 1000;
    double t = 0.0;
    while (t < T) {
        state = model.getCurrState();
        std::cout << t << "\t";
        std::cout << state[0] << "\t " << state[1]<< std::endl;
        model.run(1);
        t += dt;
        if (state[0] < -M_PI*0.5 || state[0] > 0.5*M_PI) break;
    }
}

void testModelFull(){
// dt = 0.1 is from the paper Neural Fitted Q iteration - First Experiences with..   
    double dt = 0.02;
    Model_PoleFull model(dt);
    model.createInitialState();
    State state(2);
    double T = 1000;
    double t = 0.0;
    while (t < T) {
        state = model.getCurrState();
        std::cout << t << "\t";
        std::cout << state[0] << "\t " << state[1]<< std::endl;
        model.run(2);
        t += dt;
        if (state[0] < -M_PI*0.5 || state[0] > 0.5*M_PI || state[1] < -2.4 || state[1] > 2.4) break;
    }



}
#endif
