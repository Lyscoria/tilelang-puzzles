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

extern "C" __global__ void tl_mul_relu_1d_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C);
extern "C" __global__ void __launch_bounds__(256, 1) tl_mul_relu_1d_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C) {
  for (int i_s = 0; i_s < 4; ++i_s) {
    half_t tmp = (A[(((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 4)) + i_s)] * B[(((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 4)) + i_s)]);
    half_t condval;
    if ((half_t(0x0p+0f/*0.000000e+00*/) < tmp)) {
      condval = tmp;
    } else {
      condval = half_t(0x0p+0f/*0.000000e+00*/);
    }
    C[(((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 4)) + i_s)] = condval;
  }
}