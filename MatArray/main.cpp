#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <armadillo>

#include "MatArray.h"



int main(int argc, char *argv[]) {
    MatArray<double>::Mat1DArray_ptr matArr = MatArray<double>::build(5);

    for (int i = 0 ; i < 5; i++) {
        (*matArr)[i].randu(5,5);
        (*matArr)[i].print("1D");
    }

    MatArray<double>::Mat2DArray_ptr mat2DArr = MatArray<double>::build(2,2);

    for (int i = 0 ; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            (*mat2DArr)[i][j].randu(5,5);
            (*mat2DArr)[i][j].print("2D");
        }
    }

//  here I try to test Tensor_4D
    Tensor_4D tensor(2,3,4,5);
    
    assert(2==tensor.dim1());
    assert(3==tensor.dim2());
    assert(4==tensor.dim3());
    assert(5==tensor.dim4());
    assert(120==tensor.size());
    
    tensor.fill_randn();
    tensor.print();
    tensor.fill_zeros();
    tensor.print();
    
    arma::vec v(20,arma::fill::randn);
    Tensor_4D tensor2(v.memptr(), 20, 1,1,4,5);
    assert(1==tensor2.dim1());
    assert(1==tensor2.dim2());
    assert(4==tensor2.dim3());
    assert(5==tensor2.dim4());
    assert(20==tensor2.size());
    
    v.print("arma::v");
    tensor2.print();
    
    Tensor_4D tensor3(v.memptr(), 20, 1,1,4,5,true);
    tensor3.fill_zeros();
    v.print("arma::v");
    
    tensor2.substract(tensor3,1.0);
    tensor2.print();
    
    tensor3.substract(tensor2,1.0);
    tensor3.print();
    
    Tensor_4D t4(1,2,3,4);
    int count = 0;
    for (int i = 0; i < t4.dim4(); i++){
        for (int j = 0; j < t4.dim3(); j++){
            for (int k = 0; k < t4.dim2(); k++){
                for (int m = 0; m < t4.dim1(); m++){
                    t4(m,k,j,i) = count++;
                }
            }
        
        }
    }
    
    
    arma::vec v2(t4.getPtr(),t4.size());
    
    
    t4.print();
    
    for (int i= 0; i < t4.size(); i++)
        t4(i) -= i;
    
    t4.print();
    
    v2.print("arma::v2");
    
    
    Tensor_4D t5(2,2,2,2);
    
    t5.fill_randu();
    
    
    t5.print();
    
    t5.transform([](double val){return val-0.5;});
    
    t5.print();
    
    
    
    
    
    

}

