#pragma once
#include <vector>
#include <memory>
#include <armadillo>


template<typename T>
class MatArray{	
   
    
 private:	
  MatArray(){};

	
 public:
   typedef std::vector<arma::Mat<T>> Mat1DArray;	
   typedef std::vector<std::vector<arma::Mat<T>>> Mat2DArray;
   typedef std::shared_ptr<Mat1DArray> Mat1DArray_ptr;
   typedef std::shared_ptr<Mat2DArray> Mat2DArray_ptr;
   
     
  static Mat1DArray_ptr build(int dim1){
	Mat1DArray_ptr v(new Mat1DArray);
        arma::Mat<T> m;
	for (int i = 0; i < dim1; i++)
        {   
	   v->push_back(m);
	}
           return v;	  
  }
  
  static Mat1DArray_ptr build(int dim1, int matDim1, int matDim2){
	Mat1DArray_ptr v(new Mat1DArray);
         arma::Mat<T> m(matDim1, matDim2);
	for (int i = 0; i < dim1; i++)
        {  
	   v->push_back(m);
	}
           return v;	  
  }
  static Mat2DArray_ptr build(int dim1,int dim2, int matDim1, int matDim2){
	Mat2DArray_ptr v(new Mat2DArray);
	std::vector<arma::Mat<T> > M1;
        arma::Mat<T> M2(matDim1, matDim2);
	for (int i = 0; i < dim1; i++){
            v->push_back( M1 );
            for (int j = 0; j< dim2; j++){
                (*v)[i].push_back( M2 );
            }    
	}  
	  	  
	return v;
  }
  static Mat2DArray_ptr build(int dim1, int dim2){
	Mat2DArray_ptr v(new Mat2DArray);
        std::vector<arma::Mat<T> > M1;
        arma::Mat<T> M2;	
	for (int i = 0; i < dim1; i++){
            v->push_back( M1 );
            for (int j = 0; j< dim2; j++){
                (*v)[i].push_back(M2);
            }    
	}  
	  	  
	return v;
  }
//  arma::Mat<T> & operator[](int i){return }
};