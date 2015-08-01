#pragma once
#include <memory>
#include <armadillo>

namespace NeuralNet {

    struct ElementwiseLayer {

        ElementwiseLayer() {
//            inputSize = inputSize0;
            //we only need to assign memory to the output
            delta_outOne = std::make_shared<arma::mat>();
            delta_outTwo = std::make_shared<arma::mat>();
            output = std::make_shared<arma::mat>();
        };


        std::shared_ptr<arma::mat> inputOne, inputTwo, output;
        std::shared_ptr<arma::mat> delta_outOne, delta_outTwo;
        std::vector<std::shared_ptr<arma::mat>> inputMemOne, inputMemTwo;
        void saveInputMemory();
        void activateUp();
        void updatePara(std::shared_ptr<arma::mat> delta_in, int timePoint);

//        int inputSize;
    };

}
