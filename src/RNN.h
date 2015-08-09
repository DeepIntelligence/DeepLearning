#pragma once
#include <memory>
#include <armadillo>
#include <iostream>
#include <vector>
#include "BaseLayer_LSTM.h"
#include "Net.h"
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
        std::shared_ptr<BaseLayer_LSTM> getOutputLayer(){return netOutputLayer;}
    private:
        void fillNetGradVector();
        DeepLearning::NeuralNetParameter neuralNetPara;
        double learningRate = 0.1;
        /* network gradients*/
        std::vector<std::shared_ptr<arma::mat>> netGradVector;
        std::shared_ptr<arma::mat> netOutput_;
        std::vector<BaseLayer_LSTM> hiddenLayers;
        std::shared_ptr<BaseLayer_LSTM> netOutputLayer;
        std::shared_ptr<arma::mat> trainingY, trainingX;
        int numHiddenLayers, hiddenLayerInputDim, hiddenLayerOutputDim;
        int rnnInputDim, rnnOutputDim;
    };
}


