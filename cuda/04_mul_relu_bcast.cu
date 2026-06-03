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

extern "C" __global__ void tl_mul_relu_bcast_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C);
extern "C" __global__ void __launch_bounds__(256, 1) tl_mul_relu_bcast_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C) {
  half_t A_frag[16];
  half_t B_frag[8];
  half_t C_frag[16];
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    *(uint4*)(A_frag + (i * 8)) = *(uint4*)(A + (((((((int)blockIdx.x) * 262144) + (i * 131072)) + ((((int)threadIdx.x) >> 3) * 4096)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.x) & 7) * 8)));
  }
  *(uint4*)(B_frag + 0) = *(uint4*)(B + ((((int)blockIdx.y) * 64) + ((((int)threadIdx.x) & 7) * 8)));
  #pragma unroll
  for (int i_1 = 0; i_1 < 16; ++i_1) {
    half_t res = (A_frag[i_1] * B_frag[(i_1 & 7)]);
    half_t condval;
    if ((half_t(0x0p+0f/*0.000000e+00*/) < res)) {
      condval = res;
    } else {
      condval = half_t(0x0p+0f/*0.000000e+00*/);
    }
    C_frag[i_1] = condval;
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 2; ++i_2) {
    *(uint4*)(C + (((((((int)blockIdx.x) * 262144) + (i_2 * 131072)) + ((((int)threadIdx.x) >> 3) * 4096)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.x) & 7) * 8))) = *(uint4*)(C_frag + (i_2 * 8));
  }
}