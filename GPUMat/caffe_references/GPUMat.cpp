
//copy on CPU
//copy on GPU
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

GPUMat& GPUMat::operator=(const GPUMat& rhs){
	// Check for self-assignment!
    if (this != &rhs) {
		delete _data_CPU;
		cudaFree(_data_GPU);
		_data_CPU = (double *)malloc(rhs.n_elem * sizeof(double));
		double *p = rhs.memptr();
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

	double scale = 1;
	stat=cublasDaxpy(handle,n_elem,&scale,rhs.memptr_GPU(),1,this->memptr_GPU,1);	
	return *this;
}


const GPUMat GPUMat::operator+(const GPUMat& rhs) const{
	GPUMat result = *this;     // Make a copy of myself.  Same as MyClass result(*this);
    result += other;            // Use += to add other to the copy.
    return result;              // All done!

}



