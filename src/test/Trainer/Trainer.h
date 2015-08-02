#include "common.h"
#include "../MultiLayerPerceptron/Net.h"

namespace NeuralNet {

    class Trainer {
    public:
        Trainer(const DeepLearning::NeuralNetParameter& nnPara){
            trainingParameter = nnPara;        
        }

        virtual ~Trainer() {
        }
        virtual void getGradientFromNet();
        virtual void applyUpdatesToNet();
        virtual void calUpdates() = 0;
        void setTrainingSamples(std::shared_ptr<arma::mat> X,std::shared_ptr<arma::mat> Y){
            trainingX = X;
            trainingY = Y;
        }
        void setNet(std::shared_ptr<Net> net0) {
            net = net0;
        };
        virtual void train() = 0;
    protected:
        int iter;
        std::vector<std::shared_ptr<arma::mat> > currUpdate, prevUpdate;
        std::shared_ptr<Net> net;
        DeepLearning::NeuralNetParameter trainingParameter;
    };

    class Trainer_SGD : public Trainer {
    public:
        Trainer_SGD(const DeepLearning::NeuralNetParameter& nnPara):Trainer(nnPara){}
        virtual ~Trainer_SGD() {}
        virtual void train();
        virtual void calUpdates();

    };

    class TrainerBuilder {
        inline static std::shared_ptr<Trainer> GetTrainer(const DeepLearning::NeuralNetParameter& nnPara) {
            switch (nnPara.neuralnettrainingparameter().trainertype()) {
                case DeepLearning::NeuralNetTrainingParameter_TrainerType_SGD:
                    return std::shared_ptr<Trainer>(new Trainer_SGD(nnPara));
                    break;
                case DeepLearning::NeuralNetTrainingParameter_TrainerType_iRProp:
//                    return std::shared_ptr<Trainer_SGD>(new Trainer_SGD(nnPara));
                    break;
                default: break;
            }
        }
    };
}
