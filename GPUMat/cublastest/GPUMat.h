#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"



namespace NeuralNet{
	namespace GPUMat{

// multiply the vector d_x by the scalar al and add to d_y
// d_y = al*d_x + d_y , d_x ,d_y - n- vectors ; al - scalar
//		void vec_add_vec(double *y, double *x, double scalar, int n);		
		double vec_dot_vec(double *y, double *x, int n);
//3.3.2
		void mat_prod_vec(double *mat, int mat_m, int mat_n, cublasOperation_t op, double *vec, double *result); 

		
		void mat_prod_mat(double* a, cublasOperation_t op_a, double* b, cublasOperation_t op_b, double*c, int m, int n, int k);

		
		void mat_add_mat(double* y, double* x, double scalar, int n); 

		void mat_elem_prod_mat();
				
		double mat_norm2(double *x, int n);


		
	//	template<typename Func>
	//	transform(double *x, Func func, int n);

	}
}
