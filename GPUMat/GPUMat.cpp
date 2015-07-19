#include "GPUMat.h"
#include "GPU_Math_Func.h"


GPUMat::GPUMat(int row0, int col0){
	this->n_rows = row0;
	this->n_cols = col0;
	this->n_elem = row0 * col0;
	CUDA_CHECK(cudaMalloc((void **)&_data_GPU ,this->n_elem * sizeof(*_data_GPU)));
}
#if 0
GPUMat& GPUMat::copyFromCPU(const GPUMat& rhs){
	// Check for self-assignment!
    if (this != &rhs) {
		delete _data_CPU;
		cudaFree(_data_GPU);
		cudaStat = cudaMalloc((void **)&_data_GPU ,n_elem*sizeof(double));
		cudaStat = cublasDcopy(handle, n_elem, rhs.memptr_GPU(),1, _data_GPU,1);
       // Deallocate, allocate new space, copy values...
    }
    // 1.  Deallocate any memory that MyClass is using internally
    // 2.  Allocate some memory to hold the contents of rhs
    // 3.  Copy the values from rhs into this instance
    // 4.  Return *this
	return *this;
}
#endif
GPUMat& GPUMat::operator=(const GPUMat& rhs){
	// Check for self-assignment!
    if (this != &rhs) {
		delete _data_CPU;
		cudaFree(_data_GPU);
		_data_CPU = (double *)malloc(rhs.n_elem * sizeof(double));
		const double *p = rhs.memptr_CPU();
		for (int i = 0; i < n_elem; i++) {
			_data_CPU[i] = *(p+i);		
		}
		
		this->syncToGPU();
       // Deallocate, allocate new space, copy values...
    }
    // 1.  Deallocate any memory that MyClass is using internally
    // 2.  Allocate some memory to hold the contents of rhs
    // 3.  Copy the values from rhs into this instance
    // 4.  Return *this
	return *this;
}

GPUMat& GPUMat::st(){
	this->transposeFlag=((this->transposeFlag==false)?true:false);
	return *this;
}

GPUMat& GPUMat::operator+=(const GPUMat& rhs){

	ASSERT(this->n_elem==rhs.n_elem, "number of elements not equal for addition");
	gpu_add(n_elem, this->memptr_GPU(), rhs.memptr_GPU(), this->memptr_GPU());
	return *this;
}


const GPUMat GPUMat::operator+(const GPUMat& rhs) const{
	GPUMat result = *this;     // Make a copy of myself.  Same as MyClass result(*this);
    result += rhs;            // Use += to add other to the copy.
    return result;              // All done!

}

void GPUMat::ones() {
	gpu_set(this->n_elem, 1.0, this->memptr_GPU());
}

void GPUMat::zeros(){
	gpu_set(this->n_elem, 0.0, this->memptr_GPU());
}

void GPUMat::print(std::string str) {
	this->syncToCPU();
	std::cout << str << std::endl;
	for (int i = 0; i < this->n_rows; i++){
		for (int j = 0; j < this->n_cols; j++){
			std::cout << _data_CPU[j*n_rows + i] << "\t";
	
		}
		std::cout << std::endl;

	}	
}

