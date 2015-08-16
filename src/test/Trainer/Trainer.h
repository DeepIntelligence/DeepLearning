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
    protected:
        int iter;
        double learningRate;

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
                case DeepLearning::NeuralNetTrainingParameter_TrainerType_SGDRNN:
                    return std::shared_ptr<Trainer>(new Trainer_SGDRNN(net, nnPara));
                    break;
                    
                default: break;
            }
        }
    };
}
