#pragma once
#include <memory>
#include <armadillo>
#include <iostream>
#include <vector>
#include "BaseLayer.h"
#include "RecurrLayer.h"
#include "common.h"
namespace NeuralNet {

    class RNN: public Net {
        
    public:
        RNN(DeepLearning::NeuralNetParameter);
		virtual ~RNN(){}
        
        // implementing methods required by Net interface
        virtual void forward();
        virtual void applyUpdates(std::vector<std::shared_ptr<arma::mat>>);
        virtual void calGradient();
        virtual double getLoss();
        virtual void save(std::string filename);
        virtual void load(std::string filename);
        virtual std::shared_ptr<arma::mat> netOutput();
        virtual std::shared_ptr<arma::mat> netOutputAtTime(int time);
        virtual arma::mat forwardInTime(std::shared_ptr<arma::mat> x);
        virtual void updateInternalState();
        virtual void resetWeight();
        virtual void zeroTime();
        
        
        void backward();
        void calNumericGrad();        
        void saveLayerInputOutput();
        virtual int getTime();
        virtual void setTime(int t);
        std::shared_ptr<BaseLayer> getOutputLayer(){return netOutputLayer;}
        std::vector<RecurrLayer> getHiddenLayers(){ return hiddenLayers;}
    private:
        void fillNetGradVector();
        std::shared_ptr<arma::mat> netOutput_;
        std::vector<RecurrLayer> hiddenLayers;
        std::shared_ptr<BaseLayer> netOutputLayer;
        std::vector<std::shared_ptr<arma::mat>> outputLayers_prev_output;
        int numHiddenLayers, hiddenLayerInputDim, hiddenLayerOutputDim;
        int rnnInputDim, rnnOutputDim;
        int time;
    };
}


