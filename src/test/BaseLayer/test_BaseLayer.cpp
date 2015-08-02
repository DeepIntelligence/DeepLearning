#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <armadillo>
#include "BaseLayer.h"
#include "gtest/gtest.h"
using namespace NeuralNet;


TEST(BaseLayerTest, fillBernoulli){

    BaseLayer layer(100,10,BaseLayer::sigmoid,true,0.5);
    EXPECT_EQ(layer.dropOutRate,0.5);
//    EXPECT_TRUE(layer.dropOutFlag);
    layer.B.print();
    layer.fill_Bernoulli(layer.B.memptr(),layer.B_size);
    layer.B.print();
    
}
 


int main(int argc, char *argv[]) {
    std::shared_ptr<arma::mat> trainDataX(new arma::mat);
    std::shared_ptr<arma::mat> trainDataY(new arma::mat);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

