#include<armadillo>
#include "GPUMat.h"


#define N 10
using namespace NeuralNet;


int main(){

//	arma::mat a;	
	arma::mat a(N, N, arma::fill::randu);
	arma::mat b(N, N, arma::fill::randu);
	arma::mat c = a * b;
	
	c.save("armaresult.txt",arma::raw_ascii);	


	
#if 0	
	GPUMat::mat_prod_mat(a.memptr(), CUBLAS_OP_N, b.memptr(), CUBLAS_OP_N, c.memptr(), N, N, N);	

	
	c.save("gpuresult.txt", arma::raw_ascii);
	double *aa = nullptr;
	double *bb = nullptr;
	double *cc = nullptr;

	std::swap(aa,a.memptr());
	std::swap(bb,b.memptr());
	std::swap(cc,c.memptr());
#endif
	return 0;
}
