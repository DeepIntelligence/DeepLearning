#include<armadillo>
#include <cuda_runtime.h>
#include "cublas_v2.h"
//#include "GPUMat.h"
#define N 10
void mat_prod_mat(const double* a, cublasOperation_t op_a, const double* b, cublasOperation_t op_b, double*c, int m, int n, int k);





int main(){

//	arma::mat a;	
	arma::mat a(N, N, arma::fill::randu);
	arma::mat b(N, N, arma::fill::randu);
	arma::mat c = a * b;
	
	c.save("armaresult.txt",arma::raw_ascii);	


	
#if 1	
	mat_prod_mat(a.memptr(), CUBLAS_OP_N, b.memptr(), CUBLAS_OP_N, c.memptr(), N, N, N);	

	
	c.save("gpuresult.txt", arma::raw_ascii);
//	double *aa = nullptr;
//	double *bb = nullptr;
//	double *cc = nullptr;

//	std::swap(aa,a.memptr());
//	std::swap(bb,b.memptr());
//	std::swap(cc,c.memptr());
#endif
	return 0;
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
	double bet=0.0;
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
