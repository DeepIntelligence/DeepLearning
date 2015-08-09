#include "Model_PoleSimple.h"
#include "BaseModel.h"
#include "NN_RLSolver.h"
#include "MultiLayerPerceptron.h"
#include "Util.h"
#include "common.h"
void testModel();
void testSolver(char* filename);
using namespace ReinforcementLearning;
using namespace DeepLearning;
using namespace NeuralNet;
int main(int argc, char* argv[]){

 //   for (int i = 0; i < 100; i++)
 //       testModel();
    testSolver(argv[1]);
    return 0;
}

void testSolver(char* filename){
    RL_TrainingPara RLPara;
    RLPara.numEpisodes = 300;
    RLPara.maxIter = 300;
    RLPara.discount = 0.9;
    RLPara.experienceReplayFlag = 1;
    RLPara.trainingSampleSize = 100;
    double dt = 0.1;
    NeuralNetParameter message; 
    ReadProtoFromTextFile(filename, &message);
    arma::arma_rng::set_seed_random();
    std::shared_ptr<Net> net(new MultiLayerPerceptron(message));
    std::shared_ptr<Trainer> trainer(TrainerBuilder::GetTrainer(net,message));
    std::shared_ptr<BaseModel> model(new Model_PoleSimple(dt));
    NN_RLSolver rlSolver(model, net, trainer, RLPara, 2);
    rlSolver.train();
    net->save("trainedresult");
}
void testModel(){
    std::vector<double> a;
    a.reserve(2);
    a[0] = 1;
    a[1] = 2;
    std::cout << a[0] << std::endl;
    std::cout << a[1] << std::endl;
    
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

