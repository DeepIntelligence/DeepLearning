#pragma once
#include <memory>
#include <armadillo>

namespace NeuralNet {

    struct LinearAdditionLayer {

        LinearAdditionLayer() {
//            inputSize = inputSize0;
            //we only need to assign memory to the output
//            deltaOut = std::make_shared<arma::mat>();
            output = std::make_shared<arma::mat>();
        };


        std::shared_ptr<arma::mat> inputOne, inputTwo, output;
        std::shared_ptr<arma::mat> delta_out;
        void activateUp();
        void updatePara(std::shared_ptr<arma::mat> delta_in);

//        int inputSize;
    };

}
