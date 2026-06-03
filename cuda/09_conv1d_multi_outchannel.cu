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

extern "C" __global__ void tl_conv1d_multi_outchannel_kernel(const half_t* __restrict__ K, half_t* __restrict__ O, const half_t* __restrict__ X);
extern "C" __global__ void __launch_bounds__(256, 1) tl_conv1d_multi_outchannel_kernel(const half_t* __restrict__ K, half_t* __restrict__ O, const half_t* __restrict__ X) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* X_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* workspace = ((void*)((char*)buf_dyn_shmem + 2048));
  half_t K_local[4];
  float tmp[64];
  half_t X_shared_local_cast[4];
  float O_local[64];
  half_t O_local_cast_1[4];
  half_t broadcast_var = half_t(0x0p+0f/*0.000000e+00*/);
  uint2 condval;
  if (((((((int)threadIdx.x) & 15) >> 3) + ((int)blockIdx.y)) < 4)) {
    condval = *(uint2*)(X + ((((((int)blockIdx.x) * 2048) + ((((int)threadIdx.x) >> 4) * 128)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) & 15) * 4)));
  } else {
    condval = make_uint2(__pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var));
  }
  *(uint2*)(((half_t*)X_shared) + (((int)threadIdx.x) * 4)) = condval;
  *(uint2*)(K_local + 0) = *(uint2*)(K + (((int)threadIdx.x) * 4));
  for (int j = 0; j < 32; ++j) {
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
      *(uint2*)(X_shared_local_cast + 0) = make_uint2(__pack_half2(((half_t*)X_shared)[(((i * 64) + (((int)threadIdx.x) >> 3)) + j)], ((half_t*)X_shared)[(((i * 64) + (((int)threadIdx.x) >> 3)) + j)]), __pack_half2(((half_t*)X_shared)[(((i * 64) + (((int)threadIdx.x) >> 3)) + j)], ((half_t*)X_shared)[(((i * 64) + (((int)threadIdx.x) >> 3)) + j)]));
      float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
      float4 condval_1;
      if ((((((((int)threadIdx.x) >> 3) + j) >> 5) + ((int)blockIdx.y)) < 4)) {
        float4 __1;
          float4 __2;
          uint2 v_ = *(uint2*)(X_shared_local_cast + 0);
          ((float2*)(&__2))[0] = __half22float2(((half2*)(&v_))[0]);
          ((float2*)(&__2))[1] = __half22float2(((half2*)(&v_))[1]);
          float4 __3;
          uint2 v__1 = *(uint2*)(K_local + 0);
          ((float2*)(&__3))[0] = __half22float2(((half2*)(&v__1))[0]);
          ((float2*)(&__3))[1] = __half22float2(((half2*)(&v__1))[1]);
          __1.x = (__2.x*__3.x);
          __1.y = (__2.y*__3.y);
          __1.z = (__2.z*__3.z);
          __1.w = (__2.w*__3.w);
        condval_1 = __1;
      } else {
        condval_1 = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
      }
      *(float4*)(tmp + (i * 4)) = condval_1;
    }
    __syncthreads();
    #pragma unroll
    for (int i_1 = 0; i_1 < 64; ++i_1) {
      O_local[i_1] = 0x0p+0f/*0.000000e+00*/;
      O_local[i_1] = (O_local[i_1] + tmp[i_1]);
      O_local[i_1] = tl::AllReduce<tl::SumOp, 256, 8, 0, tl::NamedBarrier<256>>::run(O_local[i_1], (&(((float*)workspace)[0])));
    }
    if ((((int)threadIdx.x) >> 3) == 0) {
      #pragma unroll
      for (int i_2 = 0; i_2 < 16; ++i_2) {
        uint2 __4;
        float4 v__2 = *(float4*)(O_local + (i_2 * 4));
        ((half2*)(&__4))[0] = __float22half2_rn(((float2*)(&v__2))[0]);
        ((half2*)(&__4))[1] = __float22half2_rn(((float2*)(&v__2))[1]);
        *(uint2*)(O_local_cast_1 + 0) = __4;
        *(uint2*)(O + (((((((int)blockIdx.x) * 65536) + (i_2 * 4096)) + (((int)blockIdx.y) * 1024)) + (j * 32)) + ((((int)threadIdx.x) & 7) * 4))) = *(uint2*)(O_local_cast_1 + 0);
      }
    }
  }
}