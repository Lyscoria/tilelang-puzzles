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

extern "C" __global__ void tl_softmax_kernel(const float* __restrict__ A, float* __restrict__ B);
extern "C" __global__ void __launch_bounds__(256, 1) tl_softmax_kernel(const float* __restrict__ A, float* __restrict__ B) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* workspace = ((void*)((char*)buf_dyn_shmem + 0));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 1024));
  float LSE[4];
  float A_local[16];
  float MAX[4];
  float EXP[16];
  float SUM[4];
  float B_local[16];
  float broadcast_var = -CUDART_INF_F;
  *(float4*)(LSE + 0) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  for (int by = 0; by < 64; ++by) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      *(float4*)(A_local + (i * 4)) = *(float4*)(A + (((((((int)blockIdx.x) * 262144) + (i * 65536)) + ((((int)threadIdx.x) >> 6) * 16384)) + (by * 256)) + ((((int)threadIdx.x) & 63) * 4)));
    }
    __syncthreads();
    #pragma unroll
    for (int i_1 = 0; i_1 < 4; ++i_1) {
      MAX[i_1] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 4; ++rv) {
        MAX[i_1] = max(MAX[i_1], A_local[((i_1 * 4) + rv)]);
      }
      MAX[i_1] = tl::AllReduce<tl::MaxOp, 64, 1, 0, tl::NamedBarrier<256>>::run(MAX[i_1], (&(((float*)workspace_1)[0])));
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 16; ++i_2) {
      EXP[i_2] = exp2f(((A_local[i_2] - MAX[(i_2 >> 2)]) * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/));
    }
    __syncthreads();
    #pragma unroll
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      SUM[i_3] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 4; ++rv_1) {
        SUM[i_3] = (SUM[i_3] + EXP[((i_3 * 4) + rv_1)]);
      }
      SUM[i_3] = tl::AllReduce<tl::SumOp, 64, 1, 0, tl::NamedBarrier<256>>::run(SUM[i_3], (&(((float*)workspace)[0])));
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 4; ++i_4) {
      LSE[i_4] = ((MAX[i_4] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/) + log2f((SUM[i_4] + exp2f((LSE[i_4] - (MAX[i_4] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/))))));
    }
  }
  for (int by_1 = 0; by_1 < 64; ++by_1) {
    #pragma unroll
    for (int i_5 = 0; i_5 < 4; ++i_5) {
      *(float4*)(A_local + (i_5 * 4)) = *(float4*)(A + (((((((int)blockIdx.x) * 262144) + (i_5 * 65536)) + ((((int)threadIdx.x) >> 6) * 16384)) + (by_1 * 256)) + ((((int)threadIdx.x) & 63) * 4)));
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 16; ++i_6) {
      B_local[i_6] = exp2f(((A_local[i_6] * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/) - LSE[(i_6 >> 2)]));
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 4; ++i_7) {
      *(float4*)(B + (((((((int)blockIdx.x) * 262144) + (i_7 * 65536)) + ((((int)threadIdx.x) >> 6) * 16384)) + (by_1 * 256)) + ((((int)threadIdx.x) & 63) * 4))) = *(float4*)(B_local + (i_7 * 4));
    }
  }
}