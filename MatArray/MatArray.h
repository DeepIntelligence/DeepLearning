#pragma once
#include <vector>
#include <memory>
#include <assert.h>
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


class Tensor_4D{
public:
    typedef std::shared_ptr<Tensor_4D> ptr;
    
	Tensor_4D(){}
	Tensor_4D(size_t d1, size_t d2, size_t d3, size_t d4){		
		_dim1 = d1;
		_dim2 = d2;
		_dim3 = d3;
		_dim4 = d4;
		_size =  _dim1 * _dim2 * _dim3 * _dim4;
		data = (double *) malloc( _size * sizeof(double));
	}
	// construct from memory, no copy or copy
	Tensor_4D(void *ptr, size_t size0, size_t d1, size_t d2, size_t d3, size_t d4, bool copyFlag = true){
		if (copyFlag) {
			data = (double *) malloc( size0 * sizeof(double));
			double *p = (double *)ptr;
			for (int i = 0; i < size0; i++)
				*(data + i) = *(p + i);
		} else {
			double *p = (double *)ptr;
			data = p;
		}
		_dim1 = d1;
		_dim2 = d2;
		_dim3 = d3;
		_dim4 = d4;
		_size =  _dim1 * _dim2 * _dim3 * _dim4;
	}
	~Tensor_4D(){ 
//            free(data);
//            data = nullptr;
        }

	const double& operator()(size_t i, size_t j, size_t k, size_t m) const{	
	
	#ifdef DEBUG
		assert(i >=0 && i < _dim1);
		assert(j >=0 && j < _dim2);
		assert(k >=0 && k < _dim3);
		assert(m >=0 && m < _dim4);
	#endif
		size_t idx = i + _dim1 * ( j + _dim2 * ( k + _dim3 * m ));
		return data[idx];
	}
	
	double& operator()(size_t i, size_t j, size_t k, size_t m) {
	#ifdef DEBUG
		assert(i >=0 && i < _dim1);
		assert(j >=0 && j < _dim2);
		assert(k >=0 && k < _dim3);
		assert(m >=0 && m < _dim4);
	#endif	
		size_t idx = i + _dim1 * ( j + _dim2 * ( k + _dim3 * m ));
		return data[idx];
	}
	
	const double& operator()(size_t i) const{ 
	#ifdef DEBUG
		assert(i >=0 && i < _size);
	#endif	
	return data[i];
	}
	
	double& operator()(size_t i) {
	#ifdef DEBUG
		assert(i >=0 && i < _size);
	#endif	
	return data[i];
	}
	
	size_t dim1() const {return _dim1;}
	size_t dim2() const {return _dim2;}
	size_t dim3() const {return _dim3;}
	size_t dim4() const {return _dim4;}
	size_t size() const {return _size;}
	void fill_randn(){
		for (int i = 0; i < _size; i++)
			data[i] = arma::randn();
	}
	void fill_randu(){
		for (int i = 0; i < _size; i++)
			data[i] = arma::randu();
	}
	void fill_zeros(){
		for (int i = 0; i < _size; i++)
			data[i] = 0;
	}
        void fill_ones(){
		for (int i = 0; i < _size; i++)
			data[i] = 1;
	}
	void print(){
		for (int i = 0; i < _size; i++)
			std::cout << data[i] << std::endl;
	}
	
	void substract(const Tensor_4D &t2, const double scale){
		assert(t2.dim1() == _dim1 && t2.dim2() == _dim2 
		&& t2.dim3() == _dim3 && t2.dim4() == _dim4 &&
		t2.size() == _size);
		
		for (int i = 0; i < _size; i++)
			data[i] -= scale * t2.data[i]; 
	}
	template <typename func_t>
        void transform(func_t op){
            for (int i = 0; i < _size; i++)
                data[i] = op(data[i]);
        }
        
	double * getPtr(){return data;}
        
        static ptr build(size_t d1, size_t d2, size_t d3, size_t d4){
            return ptr(new Tensor_4D(d1,d2,d3,d4));
        }
private:
	double *data;
	size_t _dim1,_dim2,_dim3, _dim4, _size;
};