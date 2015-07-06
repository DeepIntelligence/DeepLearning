#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <armadillo>
#include <vector>


void workOnSequenceGeneration();

int main(int argc, char *argv[]) {

    workOnSequenceGeneration();
    return 0;
}


void workOnSequenceGeneration(){
    std::shared_ptr<arma::mat> trainingY(new arma::mat);
    trainingY->load("testdata.dat",arma::raw_ascii);
    trainingY->print();
}
