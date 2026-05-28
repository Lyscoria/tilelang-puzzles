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

extern "C" __global__ void tl_mul_relu_1d_mem_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C);
extern "C" __global__ void __launch_bounds__(256, 1) tl_mul_relu_1d_mem_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C) {
  half_t A_frag[4];
  half_t B_frag[4];
  half_t C_frag[4];
  *(uint2*)(A_frag + 0) = *(uint2*)(A + ((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 4)));
  *(uint2*)(B_frag + 0) = *(uint2*)(B + ((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 4)));
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    half_t res = (A_frag[i] * B_frag[i]);
    half_t condval;
    if ((half_t(0x0p+0f/*0.000000e+00*/) < res)) {
      condval = res;
    } else {
      condval = half_t(0x0p+0f/*0.000000e+00*/);
    }
    C_frag[i] = condval;
  }
  *(uint2*)(C + ((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 4))) = *(uint2*)(C_frag + 0);
}