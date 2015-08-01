#include "Util.h"

namespace NeuralNet{
void loadData_MNIST(std::shared_ptr<arma::mat> X, std::shared_ptr<arma::mat> Y, std::string filename0){

    std::string filename_base(filename0);
    std::string filename;
    char tag[50];
    char x;
    int count;
    int numFiles = 10;
    int featSize = 28*28;
    int labelSize = 10;
    int numSamples = 1000;
    X->set_size(featSize, numFiles*numSamples);
    Y->set_size(labelSize, numFiles*numSamples);
    Y->fill(0);


    for (int i = 0 ; i < numFiles ; i++) {
        sprintf(tag,"%d",i);
        filename=filename_base+(std::string)tag;
        std::cout << filename << std::endl;
        std::ifstream infile;
        infile.open(filename,std::ios::binary | std::ios::in);
        if (infile.is_open()) {

            for (int j = 0 ; j < numSamples ; j++) {

                for (int k =0 ; k <featSize; k ++) {
                    infile.read(&x,1);
//        std::cout << x << std::endl;
                    (*X)(k, i+numFiles*j)=((unsigned char)x)/256.0;

                }
                (*Y)(i, i+numFiles*j) = 1;
//        count++;
            }

        } else {
            std::cout << "open file failure!" << std::endl;
        }

        std::cout << "dataloading finish!" <<std::endl;

    }

}
}
