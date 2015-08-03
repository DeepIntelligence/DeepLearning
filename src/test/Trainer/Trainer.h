#include "common.h"
#include "../MultiLayerPerceptron/Net.h"

namespace NeuralNet {

    class Trainer {
    public:
        Trainer(std::shared_ptr<Net> net0, const DeepLearning::NeuralNetParameter& nnPara){
            net = net0;
            trainingParameter = nnPara;
        }

        virtual ~Trainer() {
        }
        std::vector<std::shared_ptr<arma::mat>> getGradientFromNet();
        virtual void applyUpdatesToNet();
        virtual void calUpdates() = 0;
        virtual void setTrainingSamples(std::shared_ptr<arma::mat> X,std::shared_ptr<arma::mat> Y){
            trainingX = X;
            trainingY = Y;
        }
        void setNet(std::shared_ptr<Net> net0) {
            net = net0;
        };
        virtual void train() = 0;
    protected:
        int iter;
        double learningRate;
        std::shared_ptr<arma::mat> trainingX, trainingY;
        std::vector<std::shared_ptr<arma::mat>> currUpdate, prevUpdate;
        std::shared_ptr<Net> net;
        DeepLearning::NeuralNetParameter trainingParameter;
    };

    class Trainer_SGD : public Trainer {
    public:
        Trainer_SGD(std::shared_ptr<Net> net, const DeepLearning::NeuralNetParameter& nnPara):Trainer(net,nnPara){
            currUpdate = net->netGradients();
            // allocat memory for the prevUpdate
            for (int i = 0; i < currUpdate.size(); i++){
               prevUpdate.push_back(std::shared_ptr<arma::mat>(new arma::mat));
            }        
        }
        virtual ~Trainer_SGD() {}
        virtual void train();
        virtual void calUpdates();
    private:
        std::vector<std::shared_ptr<arma::mat>> prevUpdate;
    };

    
    class Trainer_iRProp : public Trainer {
    public:
        Trainer_iRProp(std::shared_ptr<Net> net, const DeepLearning::NeuralNetParameter& nnPara):Trainer(net,nnPara){
            currUpdate = net->netGradients();
            // allocat memory for the prevUpdate
            for (int i = 0; i < currUpdate.size(); i++) {
               prevUpdate.push_back(std::shared_ptr<arma::mat>(new arma::mat));
               prevDelta.push_back(std::shared_ptr<arma::mat>(new arma::mat));
               currDelta.push_back(std::shared_ptr<arma::mat>(new arma::mat));
            }
        }
        virtual ~Trainer_iRProp(){}
        virtual void train();
        virtual void calUpdates();
    private:
        std::vector<std::shared_ptr<arma::mat>> currDelta, prevDelta;
    };
    
    class TrainerBuilder {
    public:
        inline static std::shared_ptr<Trainer> GetTrainer(std::shared_ptr<Net> net, const DeepLearning::NeuralNetParameter& nnPara) {
            switch (nnPara.neuralnettrainingparameter().trainertype()) {
                case DeepLearning::NeuralNetTrainingParameter_TrainerType_SGD:
                    return std::shared_ptr<Trainer>(new Trainer_SGD(net, nnPara));
                    break;
                case DeepLearning::NeuralNetTrainingParameter_TrainerType_iRProp:
                    return std::shared_ptr<Trainer>(new Trainer_iRProp(net, nnPara));
                    break;
                default: break;
            }
        }
    };
}
