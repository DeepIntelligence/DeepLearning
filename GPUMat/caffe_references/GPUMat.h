#pragma once
#include <cuda_runtime.h>
#include "cublas_v2.h"

class GPUMat{
private:
	double* _data_CPU, _data_GPU;
	
public:
	enum MemLocation {CPU_GPU, GPU_ONLY, CPU_ONLY}
	GPUMat();
	GPUMat(int row0, int col0);
	~GPUMat(){ 
		delete _data_CPU;
		cudaFree(_data_GPU);
		cublasDestroy(handle);
	}
	void syncToGPU(){
		if(!_data_CPU) _data_CPU = (double *) malloc(n_elems * sizeof(double));
		if( loc==CPU_ONLY) loc = CPU_GPU;
		cudaStat = cudaMalloc((void **)&_data_GPU ,n_elem*sizeof(double));
		stat = cublasSetMatrix (n_rows,n_clos, sizeof(double) ,_data_CPU,n_rows,_data_GPU ,n_cols); //a -> d_a
	}
	void syncToCPU(){
		if(!_data_CPU) _data_CPU = (double *) malloc(n_elems * sizeof(double));
		if( loc==CPU_ONLY) loc = CPU_GPU;
		stat = cublasGetMatrix (n_rows, n_cols, sizeof(double) ,_data_GPU ,n_rows,_data_CPU,n_rows);
	}
	void zeros();
	void ones();
	double* memptr_CPU(){return _data_CPU;}
	double* memptr_GPU(){return _data_CPU;}
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
	
	int n_rows, n_cols, n_elems;
	bool transposeFlag;
	cudaError_t cudaStat ; // cudaMalloc status
	cublasStatus_t stat ; // CUBLAS functions status
	cublasHandle_t handle ; // CUBLAS context
}
