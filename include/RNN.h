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
//        virtual std::shared_ptr<arma::mat> netOutputAtTime(int time);
        virtual arma::mat forwardInTime(std::shared_ptr<arma::mat> x);
        virtual void updateInternalState();
        virtual void resetWeight();
        virtual void zeroTime();
        
        
        virtual void backward();
        virtual void calNumericGrad();        
        virtual void saveLayerInputOutput();
        virtual int getTime();
        virtual void setTime(int t);
        virtual BaseLayer getOutputLayer(){return baseLayers[numBaseLayers - 1];}
        virtual std::vector<RecurrLayer> getRecurrLayers(){ return recurrLayers;}
    protected:
        void fillNetGradVector();
        std::shared_ptr<arma::mat> netOutput_;
        std::vector<RecurrLayer> recurrLayers;
        std::vector<BaseLayer> baseLayers;
        int numRecurrLayers, recurrLayerInputDim, recurrLayerOutputDim, numBaseLayers;
        int rnnInputDim, rnnOutputDim;
        int time;
    };
}


