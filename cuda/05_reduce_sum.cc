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

extern "C" __global__ void tl_reduce_sum_kernel(const float* __restrict__ A, float* __restrict__ B);
extern "C" __global__ void __launch_bounds__(256, 1) tl_reduce_sum_kernel(const float* __restrict__ A, float* __restrict__ B) {
  float B_frag[2];
  float A_frag[8];
  float B_frag_clear[2];
  float broadcast_var = 0x0p+0f/*0.000000e+00*/;
  *(float2*)(B_frag + 0) = make_float2(broadcast_var, broadcast_var);
  for (int i = 0; i < 128; ++i) {
    #pragma unroll
    for (int i_1 = 0; i_1 < 2; ++i_1) {
      *(float4*)(A_frag + (i_1 * 4)) = *(float4*)(A + (((((((int)blockIdx.x) * 262144) + (i_1 * 131072)) + ((((int)threadIdx.x) >> 5) * 16384)) + (i * 128)) + ((((int)threadIdx.x) & 31) * 4)));
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 2; ++i_2) {
      B_frag_clear[i_2] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv = 0; rv < 4; ++rv) {
        B_frag_clear[i_2] = (B_frag_clear[i_2] + A_frag[((i_2 * 4) + rv)]);
      }
      B_frag_clear[i_2] = tl::AllReduce<tl::SumOp, 32, 1, 0, tl::NamedBarrier<256>>::run(B_frag_clear[i_2]);
      B_frag[i_2] = (B_frag[i_2] + B_frag_clear[i_2]);
    }
  }
  if ((((int)threadIdx.x) % 32) == 0) {
    #pragma unroll
    for (int i_3 = 0; i_3 < 2; ++i_3) {
      B[(((((int)blockIdx.x) * 16) + (i_3 * 8)) + (((int)threadIdx.x) >> 5))] = B_frag[i_3];
    }
  }
}