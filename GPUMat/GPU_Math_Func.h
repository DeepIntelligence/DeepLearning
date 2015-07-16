#pragma once
#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit
#include "cblas.h"
#include "device_common.h"



// Decaf gpu gemm provides an interface that is almost the same as the cpu
// gemm function - following the c convention and calling the fortran-order
// gpu code under the hood.
void gpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C);

void gpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
    const double alpha, const double* A, const double* x, const double beta,
    double* y);

void gpu_axpy(const int N, const double alpha, const double* X,
    double* Y);

void gpu_axpby(const int N, const double alpha, const double* X,
    const double beta, double* Y);

void gpu_memcpy(const size_t N, const void *X, void *Y);

void gpu_set(const int N, const double alpha, double *X);

inline void gpu_memset(const size_t N, const int alpha, void* X) {
  CUDA_CHECK(cudaMemset(X, alpha, N));  // NOLINT(caffe/alt_fn)
}

void gpu_add_scalar(const int N, const double alpha, double *X);

void gpu_scal(const int N, const double alpha, double *X);

void gpu_add(const int N, const double* a, const double* b, double* y);

void gpu_sub(const int N, const double* a, const double* b, double* y);

void gpu_mul(const int N, const double* a, const double* b, double* y);

void gpu_abs(const int n, const double* a, double* y);

template<typename functor>
void gpu_transform(const int n, const double* a, functor y);

// gpu_rng_uniform with two arguments generates integers in the range
// [0, UINT_MAX].
void gpu_rng_uniform(const int n, unsigned int* r);

// gpu_rng_uniform with four arguments generates floats in the range
// (a, b] (strictly greater than a, less than or equal to b) due to the
// specification of curandGenerateUniform.  With a = 0, b = 1, just calls
// curandGenerateUniform; with other limits will shift and scale the outputs
// appropriately after calling curandGenerateUniform.

void gpu_rng_uniform(const int n, const double a, const double b, double* r);

void gpu_rng_gaussian(const int n, const double mu, const double sigma,
                            double* r);

void gpu_rng_bernoulli(const int n, const double p, int* r);

void gpu_dot(const int n, const double* x, const double* y, double* out);

