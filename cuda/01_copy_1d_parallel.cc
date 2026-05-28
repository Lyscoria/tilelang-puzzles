#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void tl_copy_1d_parallel_kernel(const half_t* __restrict__ A, half_t* __restrict__ B);
extern "C" __global__ void __launch_bounds__(128, 1) tl_copy_1d_parallel_kernel(const half_t* __restrict__ A, half_t* __restrict__ B) {
  *(uint4*)(B + ((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 8))) = *(uint4*)(A + ((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 8)));
}