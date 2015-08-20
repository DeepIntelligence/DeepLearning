#pragma once
#include "common.h"
#include "Net.h"

namespace NeuralNet {

    class Trainer {
    public:
        Trainer(){}
        Trainer(std::shared_ptr<Net> net0, const DeepLearning::NeuralNetParameter& nnPara){
            net = net0;
            trainingParameter = nnPara;
        }

        virtual ~Trainer() {
        }
        std::vector<std::shared_ptr<arma::mat>> getGradientFromNet();
        virtual void applyUpdatesToNet(std::vector<std::shared_ptr<arma::mat>> update);
        virtual void calUpdates() = 0;
        virtual void setTrainingSamples(std::shared_ptr<arma::mat> X,std::shared_ptr<arma::mat> Y){
            trainingX = X;
            trainingY = Y;
        }
        virtual void setTrainingSamples(std::vector<std::shared_ptr<arma::mat>> X, 
                                        std::vector<std::shared_ptr<arma::mat>> Y){
            trainingXVec = X;
            trainingYVec = Y;
        }
        void setNet(std::shared_ptr<Net> net0) {
            net = net0;
        };
        virtual void train() = 0;
        virtual void trainHelper(std::shared_ptr<arma::mat> X, 
                                 std::shared_ptr<arma::mat> Y);
        virtual void printInfo();
        virtual void printGradNorm();
    protected:
        int iter;
        double learningRate, errorTotal;

        std::shared_ptr<arma::mat> trainingX, trainingY;
        std::vector<std::shared_ptr<arma::mat>> trainingXVec, trainingYVec;

        std::vector<std::shared_ptr<arma::mat>> currUpdate, prevUpdate;
        std::shared_ptr<Net> net;
        DeepLearning::NeuralNetParameter trainingParameter;
    };

    class Trainer_SGD : public Trainer {
    public:
        Trainer_SGD(){}
        Trainer_SGD(std::shared_ptr<Net> net, const DeepLearning::NeuralNetParameter& nnPara):Trainer(net,nnPara){
            this->allocateMemory();
        }
        virtual ~Trainer_SGD() {}
        virtual void train();
        virtual void calUpdates();
        virtual void allocateMemory(){
            currUpdate = net->netGradients();
            // allocat memory for the prevUpdate
            for (int i = 0; i < currUpdate.size(); i++){
               prevUpdate.push_back(std::shared_ptr<arma::mat>(new arma::mat));
            }        
        }
    private:
        std::vector<std::shared_ptr<arma::mat>> prevUpdate;
    };
    
    class Trainer_SGDRNN : public Trainer_SGD {
        public:
        Trainer_SGDRNN(std::shared_ptr<Net> net0, const DeepLearning::NeuralNetParameter& nnPara){
            net = net0;
            trainingParameter = nnPara;
            this->allocateMemory();
        }
        virtual ~Trainer_SGDRNN() {}
        virtual void train();
        virtual void calUpdates();
        virtual void gradientClear(){
            for (int i = 0; i < currUpdate_accu.size(); i++){
                currUpdate_accu[i]->zeros();         
            }
        }
        virtual void gradientAccu(std::vector<std::shared_ptr<arma::mat>> curr){
            for (int i = 0; i < currUpdate_accu.size(); i++){
                *(currUpdate_accu[i]) += *(curr[i]);         
            }
        }
        virtual void allocateMemory(){
                currUpdate = net->netGradients();
            // allocat memory for the prevUpdate
            for (int i = 0; i < currUpdate.size(); i++){
                prevUpdate_accu.push_back(std::shared_ptr<arma::mat>(new arma::mat));
                currUpdate_accu.push_back(std::shared_ptr<arma::mat>(new arma::mat));         
            }        
        }    
        private:
            std::vector<std::shared_ptr<arma::mat>> currUpdate_accu;
            std::vector<std::shared_ptr<arma::mat>> prevUpdate_accu;
            
    };
    
        
    class Trainer_RMSProp : public Trainer {
    public:
        Trainer_RMSProp(std::shared_ptr<Net> net, const DeepLearning::NeuralNetParameter& nnPara):Trainer(net,nnPara){
            this->epsilon = this->trainingParameter.neuralnettrainingparameter().epi();
            this->rho = this->trainingParameter.neuralnettrainingparameter().rmsprop_rho();
            currUpdate = net->netGradients();
            // allocat memory
            for (int i = 0; i < currUpdate.size(); i++) {
               squared_accu.push_back(std::shared_ptr<arma::mat>(new arma::mat));
               squared_accu[i]->zeros(currUpdate[i]->n_rows, currUpdate[i]->n_cols);
            }
        }
        virtual ~Trainer_RMSProp(){}
        virtual void train();
        virtual void calUpdates();
    private:
        double rho, epsilon;
        std::vector<std::shared_ptr<arma::mat>> squared_accu;
        
        
    };
    
    class TrainerBuilder {
    public:
        inline static std::shared_ptr<Trainer> GetTrainer(std::shared_ptr<Net> net, const DeepLearning::NeuralNetParameter& nnPara) {
            switch (nnPara.neuralnettrainingparameter().trainertype()) {
                case DeepLearning::NeuralNetTrainingParameter_TrainerType_SGD:
                    return std::shared_ptr<Trainer>(new Trainer_SGD(net, nnPara));
                    break;
                case DeepLearning::NeuralNetTrainingParameter_TrainerType_RMSProp:
                    return std::shared_ptr<Trainer>(new Trainer_RMSProp(net, nnPara));
                    break;
                case DeepLearning::NeuralNetTrainingParameter_TrainerType_SGDRNN:
                    return std::shared_ptr<Trainer>(new Trainer_SGDRNN(net, nnPara));
                    break;
                    
                default: break;
            }
        }
    };
}
