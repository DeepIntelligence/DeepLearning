#include <vector>
#include <armadillo>


template<typename T>
class MatArray{	
   
    
 private:	
  MatArray(){};

	
 public:
   typedef std::vector<arma::Mat<T>> Mat1DArray;	
    typedef std::vector<std::vector<arma::Mat<T>>> Mat2DArray;       
     
  static Mat1DArray build(int dim1){
	Mat1DArray v;
	for (int i = 0; i < dim1; i++)
        {   arma::Mat<T> m;
	   v.push_back(m);
	}
           return v;	  
  }
  
  static Mat2DArray build(int dim1,int dim2){
	Mat2DArray v;
	
	for (int i = 0; i < dim1; i++){
		v.push_back(build(dim2));	
	}  
	  	  
	return v;
  }
  
//  arma::Mat<T> & operator[](int i){return }
};