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

extern "C" __global__ void tl_gemv_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C);
extern "C" __global__ void __launch_bounds__(128, 1) tl_gemv_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C) {
  float C_local[4];
  half_t A_local[32];
  half_t B_local[8];
  float AB_temp[32];
  float C_local_clear[4];
  float broadcast_var = 0x0p+0f/*0.000000e+00*/;
  *(float4*)(C_local + 0) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  for (int by = 0; by < 128; ++by) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      *(uint4*)(A_local + (i * 8)) = *(uint4*)(A + (((((((int)blockIdx.x) * 524288) + (i * 131072)) + ((((int)threadIdx.x) >> 2) * 4096)) + (by * 32)) + ((((int)threadIdx.x) & 3) * 8)));
    }
    *(uint4*)(B_local + 0) = *(uint4*)(B + ((by * 32) + ((((int)threadIdx.x) & 3) * 8)));
    #pragma unroll
    for (int i_1 = 0; i_1 < 8; ++i_1) {
      float4 __1;
        float4 __2;
        uint2 v_ = *(uint2*)(A_local + (i_1 * 4));
        ((float2*)(&__2))[0] = __half22float2(((half2*)(&v_))[0]);
        ((float2*)(&__2))[1] = __half22float2(((half2*)(&v_))[1]);
        float4 __3;
        uint2 v__1 = *(uint2*)(B_local + ((i_1 & 1) * 4));
        ((float2*)(&__3))[0] = __half22float2(((half2*)(&v__1))[0]);
        ((float2*)(&__3))[1] = __half22float2(((half2*)(&v__1))[1]);
        __1.x = (__2.x*__3.x);
        __1.y = (__2.y*__3.y);
        __1.z = (__2.z*__3.z);
        __1.w = (__2.w*__3.w);
      *(float4*)(AB_temp + (i_1 * 4)) = __1;
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 4; ++i_2) {
      C_local_clear[i_2] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv = 0; rv < 8; ++rv) {
        C_local_clear[i_2] = (C_local_clear[i_2] + AB_temp[((i_2 * 8) + rv)]);
      }
      C_local_clear[i_2] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(C_local_clear[i_2]);
      C_local[i_2] = (C_local[i_2] + C_local_clear[i_2]);
    }
  }
  if ((((int)threadIdx.x) % 4) == 0) {
    #pragma unroll
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      C[(((((int)blockIdx.x) * 128) + (i_3 * 32)) + (((int)threadIdx.x) >> 2))] = ((half_t)C_local[i_3]);
    }
  }
}