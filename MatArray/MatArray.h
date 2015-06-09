#pragma once
#include <vector>
#include <memory>
#include <armadillo>


template<typename T>
class MatArray {


private:
    MatArray() {};


public:
    typedef std::vector<arma::Mat<T>> Mat1DArray;
    typedef std::vector<std::vector<arma::Mat<T>>> Mat2DArray;
    typedef std::shared_ptr<Mat1DArray> Mat1DArray_ptr;
    typedef std::shared_ptr<Mat2DArray> Mat2DArray_ptr;


    static Mat1DArray_ptr build(int dim1) {
        Mat1DArray_ptr v(new Mat1DArray);
        arma::Mat<T> m;
        for (int i = 0; i < dim1; i++) {
            v->push_back(m);
        }
        return v;
    }

    static Mat1DArray_ptr build(int dim1, int matDim1, int matDim2) {
        Mat1DArray_ptr v(new Mat1DArray);
        arma::Mat<T> m(matDim1, matDim2);
        for (int i = 0; i < dim1; i++) {
            v->push_back(m);
        }
        return v;
    }
    static Mat2DArray_ptr build(int dim1,int dim2, int matDim1, int matDim2) {
        Mat2DArray_ptr v(new Mat2DArray);
        std::vector<arma::Mat<T> > M1;
        arma::Mat<T> M2(matDim1, matDim2);
        for (int i = 0; i < dim1; i++) {
            v->push_back( M1 );
            for (int j = 0; j< dim2; j++) {
                (*v)[i].push_back( M2 );
            }
        }

        return v;
    }
    static Mat2DArray_ptr build(int dim1, int dim2) {
        Mat2DArray_ptr v(new Mat2DArray);
        std::vector<arma::Mat<T> > M1;
        arma::Mat<T> M2;
        for (int i = 0; i < dim1; i++) {
            v->push_back( M1 );
            for (int j = 0; j< dim2; j++) {
                (*v)[i].push_back(M2);
            }
        }

        return v;
    }
    
  	 static void fillZeros(Mat2DArray_ptr p){
  	 	  int n_row = (*p).size();
  	 	  int n_col = (*p)[0].size();
  	 	  for (int i = 0; i < n_row; i++){
  	 	  		for (int j = 0; j < n_col; j++){
  	 	  				(*p)[i][j].zeros();
  	 	  		
  	 	  		}
  	 	  }
  	 }
  	 
  	 static void substract(Mat2DArray_ptr p1, Mat2DArray_ptr p2, double scale){
  	 	  int n_row = (*p1).size();
  	 	  int n_col = (*p1)[0].size();
  	 	  for (int i = 0; i < n_row; i++){
  	 	  		for (int j = 0; j < n_col; j++){
  	 	  				(*p1)[i][j] -= ((*p2)[i][j] * scale);
  	 	  		
  	 	  		}
  	 	  }  	 		
  	 
  	 
  	 }  
         
    static void save(Mat2DArray_ptr p,std::string filenamebase){
        char tag[50];
        std::string filename;
        int n_row = (*p).size();
  	int n_col = (*p)[0].size();
  	for (int i = 0; i < n_row; i++){
            for (int j = 0; j < n_col; j++){
                sprintf(tag,"_%d_%d",i,j);
                filename = filenamebase + (std::string)tag;
  		(*p)[i][j].save(filename,arma::raw_ascii);
            }
  	}  
    }
};
