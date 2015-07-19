#include "device_common.h"

std::shared_ptr<GPUEnv> GPUEnv::singleton_;

GPUEnv::GPUEnv()
    : cublas_handle_(NULL), curand_generator_(NULL){
  // Try to create a cublas handler, and report an error if failed (but we will
  // keep the program running as one might just want to run CPU code).
  if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "Cannot create Cublas handle. Cublas won't be available.";
  }
#if 0  
  // Try to create a curand handler.
  if (curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT)
      != CURAND_STATUS_SUCCESS ||
      curandSetPseudoRandomGeneratorSeed(curand_generator_, cluster_seedgen())
      != CURAND_STATUS_SUCCESS) {
    std::cerr << "Cannot create Curand generator. Curand won't be available.";
  }
#endif  
}

GPUEnv::~GPUEnv() {
  if (cublas_handle_) CUBLAS_CHECK(cublasDestroy(cublas_handle_));
#if 0
  if (curand_generator_) {
    CURAND_CHECK(curandDestroyGenerator(curand_generator_));
  }
#endif  
}


void GPUEnv::DeviceQuery() {
  cudaDeviceProp prop;
  int device;
  if (cudaSuccess != cudaGetDevice(&device)) {
    printf("No cuda device present.\n");
    return;
  }
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  ofstream os;
  os.open("GPU_info.log");
  os << "Device id:                     " << device;
  os << "Major revision number:         " << prop.major;
  os << "Minor revision number:         " << prop.minor;
  os << "Name:                          " << prop.name;
  os << "Total global memory:           " << prop.totalGlobalMem;
  os << "Total shared memory per block: " << prop.sharedMemPerBlock;
  os << "Total registers per block:     " << prop.regsPerBlock;
  os << "Warp size:                     " << prop.warpSize;
  os << "Maximum memory pitch:          " << prop.memPitch;
  os << "Maximum threads per block:     " << prop.maxThreadsPerBlock;
  os << "Maximum dimension of block:    "
      << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", "
      << prop.maxThreadsDim[2];
  os << "Maximum dimension of grid:     "
      << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", "
      << prop.maxGridSize[2];
  os << "Clock rate:                    " << prop.clockRate;
  os << "Total constant memory:         " << prop.totalConstMem;
  os << "Texture alignment:             " << prop.textureAlignment;
  os << "Concurrent copy and execution: "
      << (prop.deviceOverlap ? "Yes" : "No");
  os << "Number of multiprocessors:     " << prop.multiProcessorCount;
  os << "Kernel execution timeout:      "
      << (prop.kernelExecTimeoutEnabled ? "Yes" : "No");
  return;
}

#if 0
void GPUEnv::set_random_seed(const unsigned int seed) {
  // Curand seed
  static bool g_curand_availability_logged = false;
  if (Get().curand_generator_) {
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curand_generator(),
        seed));
    CURAND_CHECK(curandSetGeneratorOffset(curand_generator(), 0));
  } else {
    if (!g_curand_availability_logged) {
        std::cerr <<
            "Curand not available. Skipping setting the curand seed.";
        g_curand_availability_logged = true;
    }
  }
  // RNG seed
  Get().random_generator_.reset(new RNG(seed));
}

class GPUEnv::RNG::Generator {
 public:
  Generator() : rng_(new GPUEnv::rng_t(cluster_seedgen())) {}
  explicit Generator(unsigned int seed) : rng_(new GPUEnv::rng_t(seed)) {}
  GPUEnv::rng_t* rng() { return rng_.get(); }
 private:
  shared_ptr<GPUEnv::rng_t> rng_;
};

GPUEnv::RNG::RNG() : generator_(new Generator()) { }

GPUEnv::RNG::RNG(unsigned int seed) : generator_(new Generator(seed)) { }

GPUEnv::RNG& GPUEnv::RNG::operator=(const RNG& other) {
  generator_.reset(other.generator_.get());
  return *this;
}

void* GPUEnv::RNG::generator() {
  return static_cast<void*>(generator_->rng());
}
#endif
