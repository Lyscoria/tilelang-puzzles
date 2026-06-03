#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <math_constants.h>
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void tl_scalar_flash_attn_kernel(const float* __restrict__ K, float* __restrict__ O, const float* __restrict__ Q, const float* __restrict__ V);
extern "C" __global__ void __launch_bounds__(256, 1) tl_scalar_flash_attn_kernel(const float* __restrict__ K, float* __restrict__ O, const float* __restrict__ Q, const float* __restrict__ V) {
  float LSE[2];
  float Q_local[8];
  float K_local[8];
  float QK_local[8];
  float MAX[2];
  float EXP[8];
  float SUM[2];
  float V_local[8];
  float O_local[8];
  float broadcast_var = -CUDART_INF_F;
  *(float2*)(LSE + 0) = make_float2(broadcast_var, broadcast_var);
  for (int by = 0; by < 128; ++by) {
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
      *(float4*)(Q_local + (i * 4)) = *(float4*)(Q + (((((((int)blockIdx.x) * 262144) + (i * 131072)) + ((((int)threadIdx.x) >> 5) * 16384)) + (by * 128)) + ((((int)threadIdx.x) & 31) * 4)));
    }
    #pragma unroll
    for (int i_1 = 0; i_1 < 2; ++i_1) {
      *(float4*)(K_local + (i_1 * 4)) = *(float4*)(K + (((((((int)blockIdx.x) * 262144) + (i_1 * 131072)) + ((((int)threadIdx.x) >> 5) * 16384)) + (by * 128)) + ((((int)threadIdx.x) & 31) * 4)));
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 8; ++i_2) {
      QK_local[i_2] = (Q_local[i_2] * K_local[i_2]);
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 2; ++i_3) {
      MAX[i_3] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 4; ++rv) {
        MAX[i_3] = max(MAX[i_3], QK_local[((i_3 * 4) + rv)]);
      }
      MAX[i_3] = tl::AllReduce<tl::MaxOp, 32, 1, 0, tl::NamedBarrier<256>>::run(MAX[i_3]);
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 8; ++i_4) {
      EXP[i_4] = exp2f(((QK_local[i_4] - MAX[(i_4 >> 2)]) * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/));
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 2; ++i_5) {
      SUM[i_5] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 4; ++rv_1) {
        SUM[i_5] = (SUM[i_5] + EXP[((i_5 * 4) + rv_1)]);
      }
      SUM[i_5] = tl::AllReduce<tl::SumOp, 32, 1, 0, tl::NamedBarrier<256>>::run(SUM[i_5]);
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 2; ++i_6) {
      LSE[i_6] = ((MAX[i_6] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/) + log2f((SUM[i_6] + exp2f((LSE[i_6] - (MAX[i_6] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/))))));
    }
  }
  for (int by_1 = 0; by_1 < 128; ++by_1) {
    #pragma unroll
    for (int i_7 = 0; i_7 < 2; ++i_7) {
      *(float4*)(Q_local + (i_7 * 4)) = *(float4*)(Q + (((((((int)blockIdx.x) * 262144) + (i_7 * 131072)) + ((((int)threadIdx.x) >> 5) * 16384)) + (by_1 * 128)) + ((((int)threadIdx.x) & 31) * 4)));
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 2; ++i_8) {
      *(float4*)(K_local + (i_8 * 4)) = *(float4*)(K + (((((((int)blockIdx.x) * 262144) + (i_8 * 131072)) + ((((int)threadIdx.x) >> 5) * 16384)) + (by_1 * 128)) + ((((int)threadIdx.x) & 31) * 4)));
    }
    #pragma unroll
    for (int i_9 = 0; i_9 < 2; ++i_9) {
      *(float4*)(V_local + (i_9 * 4)) = *(float4*)(V + (((((((int)blockIdx.x) * 262144) + (i_9 * 131072)) + ((((int)threadIdx.x) >> 5) * 16384)) + (by_1 * 128)) + ((((int)threadIdx.x) & 31) * 4)));
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 8; ++i_10) {
      O_local[i_10] = (exp2f((((Q_local[i_10] * K_local[i_10]) * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/) - LSE[(i_10 >> 2)])) * V_local[i_10]);
    }
    #pragma unroll
    for (int i_11 = 0; i_11 < 2; ++i_11) {
      *(float4*)(O + (((((((int)blockIdx.x) * 262144) + (i_11 * 131072)) + ((((int)threadIdx.x) >> 5) * 16384)) + (by_1 * 128)) + ((((int)threadIdx.x) & 31) * 4))) = *(float4*)(O_local + (i_11 * 4));
    }
  }
}