#pragma once
#include <memory>
#include <armadillo>
#include <iostream>
#include <vector>
#include "BaseLayer.h"
#include "MultiAddLayer.h"
#include "common.h"
namespace NeuralNet {

    class RNN: public Net {
        
    public:
        RNN(int numHiddenLayers0, int hiddenLayerInputDim0,
        int hiddenLayerOutputDim0, int inputDim0, int outputDim0, 
        std::shared_ptr<arma::mat> trainingX0, std::shared_ptr<arma::mat> trainingY0);
        RNN(DeepLearning::NeuralNetParameter);
        
        void backward();
        void updatePara();
        void train();
        void test();
        void calNumericGrad();
        
        // implementing methods required by Net interface
        virtual void forward();
        virtual void setTrainingSamples(std::shared_ptr<arma::mat> X, std::shared_ptr<arma::mat> Y);
        virtual void applyUpdates(std::vector<std::shared_ptr<arma::mat>>);
        virtual void calGradient();
        virtual std::vector<std::shared_ptr<arma::mat>> netGradients();
        virtual double getLoss();
        virtual void save(std::string filename);
        virtual void load(std::string filename);
        virtual std::shared_ptr<arma::mat> netOutput();
        virtual std::shared_ptr<arma::mat> netOutputAtTime(int time);
        virtual arma::mat forwardInTime(std::shared_ptr<arma::mat> x);
        virtual int getTime();
        virtual void setTime(int t);
        virtual void updateInternalState();
        virtual void saveLayerInputOutput();
        std::shared_ptr<BaseLayer> getOutputLayer(){return netOutputLayer;}
    private:
        void fillNetGradVector();
        DeepLearning::NeuralNetParameter neuralNetPara;
        double learningRate = 0.1;
        /* network gradients*/
        std::vector<std::shared_ptr<arma::mat>> netGradVector;
        std::shared_ptr<arma::mat> netOutput_;
        std::vector<MultiAddLayer> hiddenLayers;
        std::shared_ptr<BaseLayer> netOutputLayer;
        std::shared_ptr<arma::mat> trainingY, trainingX;
        std::vector<std::shared_ptr<arma::mat>> outputLayers_prev_output;
        int numHiddenLayers, hiddenLayerInputDim, hiddenLayerOutputDim;
        int rnnInputDim, rnnOutputDim;
        int time;
    };
}


