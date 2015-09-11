#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

double vec_dot_vec(double *y, double *x, int n){


return 0;
}

void mat_add_mat(const double *x, double *y, double scalar, int n){
	cudaError_t cudaStat ; // cudaMalloc status
	cublasStatus_t stat ; // CUBLAS functions status
	cublasHandle_t handle ; // CUBLAS context	
// on the device
	double *d_x; // d_x - x on the device
	double *d_y; // d_y - y on the device

	cudaStat = cudaMalloc (( void **)& d_x, n*sizeof(*x)); // device
	// memory alloc for x
	cudaStat = cudaMalloc (( void **)& d_y, n*sizeof(*y)); // device
	// memory alloc for y
	stat = cublasCreate (& handle ); // initialize CUBLAS context
	stat = cublasSetVector (n, sizeof (*x), x ,1 ,d_x, 1); // cp x- >d_x
	stat = cublasSetVector (n, sizeof (*y), y ,1 ,d_y, 1); // cp y- >d_y

	stat=cublasDaxpy(handle,n,&scalar,d_x,1,d_y,1);


	cudaFree (d_x ); // free device memory
	cudaFree (d_y ); // free device memory
	cublasDestroy ( handle ); // destroy CUBLAS context

}


void mat_prod_mat(const double* a, cublasOperation_t op_a, const double* b, cublasOperation_t op_b, double*c, int m, int n, int k){

	cudaError_t cudaStat ; // cudaMalloc status
	cublasStatus_t stat ; // CUBLAS functions status
	cublasHandle_t handle ; // CUBLAS context

	// on the device
	double* d_a; // d_a - a on the device
	double* d_b; // d_b - b on the device
	double* d_c; // d_c - c on the device
	cudaStat = cudaMalloc((void **)&d_a ,m*k*sizeof(*a)); // device
	// memory alloc for a
	cudaStat = cudaMalloc((void **)&d_b ,k*n*sizeof(*b)); // device
	// memory alloc for b
	cudaStat = cudaMalloc((void **)&d_c ,m*n*sizeof(*c)); // device
	// memory alloc for c
	stat = cublasCreate(&handle); // initialize CUBLAS context
// copy matrices from the host to the device
	stat = cublasSetMatrix (m,k, sizeof(*a) ,a,m,d_a ,m); //a -> d_a
	stat = cublasSetMatrix (k,n, sizeof(*b) ,b,k,d_b ,k); //b -> d_b
	stat = cublasSetMatrix (m,n, sizeof(*c) ,c,m,d_c ,m); //c -> d_c
	double al=1.0;	
	double bet=1.0;
// matrix - matrix multiplication : d_c = al*d_a *d_b + bet *d_c
// d_a -mxk matrix , d_b -kxn matrix , d_c -mxn matrix ;
// al ,bet -scalars
	stat=cublasDgemm(handle,op_a,op_b,m,n,k,&al,d_a,m,d_b,k,&bet,d_c,m);
	
	stat = cublasGetMatrix (m, n, sizeof(*c) ,d_c ,m,c,m); // cp d_c - >c

	cudaFree (d_a ); // free device memory
	cudaFree (d_b ); // free device memory
	cudaFree (d_c ); // free device memory
	cublasDestroy ( handle ); // destroy CUBLAS context

}
