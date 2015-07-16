#pragma once
#include <memory>
#include <string>
#include <cstdlib>
#include <cuda_runtime.h>
#include "cublas_v2.h"
/* notes on the design of the GPUMat 
	1) the synchronization to the CPU is lazy
	
*/
class GPUMat{
private:
	double *_data_CPU, *_data_GPU;
	
public:
	enum MemLocation {CPU_GPU, GPU_ONLY, CPU_ONLY};
	GPUMat(){}
	GPUMat(int row0, int col0);
	~GPUMat(){ 
		delete _data_CPU;
		cudaFree((void *)_data_GPU);
//		cublasDestroy(handle);
	}
	void syncToGPU(){
//		if(!_data_CPU) _data_CPU = (double *) malloc(n_elem * sizeof(double));
//		if( loc==CPU_ONLY) loc = CPU_GPU;
		cudaMalloc((void **)&_data_GPU ,n_elem*sizeof(double));
		cublasSetMatrix (n_rows,n_cols, sizeof(double) ,_data_CPU,n_rows,_data_GPU ,n_cols); //a -> d_a
	}
	void syncToCPU(){
//		if(!_data_CPU) _data_CPU = (double *) malloc(n_elem * sizeof(double));
//		if( loc==CPU_ONLY) loc = CPU_GPU;
	_data_CPU = (double *) malloc(n_elem * sizeof(double));
		cublasGetMatrix (n_rows, n_cols, sizeof(double) ,_data_GPU ,n_rows,_data_CPU,n_rows);
	}
	void zeros();
	void ones();
	double* memptr_CPU(){return _data_CPU;}
	double* memptr_GPU(){return _data_CPU;}
	double* memptr_CPU() const {return _data_CPU;}
	double* memptr_GPU() const {return _data_CPU;}	
	GPUMat& st();
	GPUMat& operator=(const GPUMat& rhs);
	GPUMat& operator+=(const GPUMat& rhs);
	GPUMat& operator-=(const GPUMat& rhs);
	GPUMat& operator*=(const GPUMat& rhs);
	GPUMat& operator*=(const double scal);
	GPUMat& operator%=(const GPUMat& rhs);
	const GPUMat operator+(const GPUMat& rhs) const;
	const GPUMat operator-(const GPUMat& rhs) const;
	const GPUMat operator*(const GPUMat& rhs) const;
	const GPUMat operator%(const GPUMat& rhs) const;
	void print(std::string str="") const;
	
	int n_rows, n_cols, n_elem;
	bool transposeFlag;
	MemLocation loc;
//	cudaError_t cudaStat ; // cudaMalloc status
//	cublasStatus_t stat ; // CUBLAS functions status
//	cublasHandle_t handle ; // CUBLAS context
};
