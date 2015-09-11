#pragma once
#include <armadillo>
#include "common.h"
#include "BaseLayer.h"
#include "optimization.h"
#include "DeepLearning.pb.h"
#include "Net.h"
namespace NeuralNet {

    class MultiLayerPerceptron : public Net {
    public:
        MultiLayerPerceptron(DeepLearning::NeuralNetParameter);
        virtual ~MultiLayerPerceptron() {
        }
        void train();
        void initialize();
        /* forward pass*/
        void feedForward(std::shared_ptr<arma::mat>);
        /* back propogate the error to update the parameters*/
        void backProp(std::shared_ptr<arma::mat>, double learningRate);
        void backProp(std::shared_ptr<arma::mat>);
        void test(std::shared_ptr<arma::mat> trainingX, std::shared_ptr<arma::mat> trainingY);
        /* calculate the numerical gradient for testing*/
        void calNumericGrad(std::shared_ptr<arma::mat> trainingX, std::shared_ptr<arma::mat> trainingY);
        void vectoriseGrad(arma::vec &grad);
        void deVectoriseWeight(arma::vec &x);
        void vectoriseWeight(arma::vec &x);
        void calLoss(std::shared_ptr<arma::mat> delta);
        virtual void forward();
       	virtual void applyUpdates(std::vector<std::shared_ptr<arma::mat>>);
        virtual void calGradient();
        virtual double getLoss();
        virtual void save(std::string filename);
        virtual void load(std::string filename);
        virtual std::shared_ptr<arma::mat> netOutput() {
            return netOutput_;
        }
    private:
        int numLayers;
        int numInstance;
        bool testGrad;
        double error;
        /**the collection of Base layers*/
        std::vector<BaseLayer> layers;
        /* dimension parameters for each layer*/
        std::vector<int> dimensions;
        /* network output*/
        std::shared_ptr<arma::mat> netOutput_;
        int totalDim;

    };

    class MLPTrainer : public Optimization::ObjectFunc {
    public:
        MLPTrainer(MultiLayerPerceptron &MLP);

        ~MLPTrainer() {
        }
        virtual double operator()(arma::vec &x, arma::vec &grad);
        //    std::shared_ptr<arma::vec> x_init;
    private:
        MultiLayerPerceptron &MLP;
    };
}
