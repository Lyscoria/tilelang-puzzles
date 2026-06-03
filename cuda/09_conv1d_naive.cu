2026-06-03 11:35:12  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:129): TileLang begins to compile kernel `tl_conv1d_naive` with `out_idx=[-1]`
2026-06-03 11:35:16  [TileLang:tilelang.jit.kernel:INFO] (kernel.py:137): TileLang completes to compile kernel `tl_conv1d_naive`
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

extern "C" __global__ void tl_conv1d_naive_kernel(const half_t* __restrict__ K, half_t* __restrict__ O, const half_t* __restrict__ X);
extern "C" __global__ void __launch_bounds__(256, 1) tl_conv1d_naive_kernel(const half_t* __restrict__ K, half_t* __restrict__ O, const half_t* __restrict__ X) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* X_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* O_local = ((void*)((char*)buf_dyn_shmem + 2048));
  half_t K_local[1];
  float tmp[2];
  float O_local_frag[2];
  half_t broadcast_var = half_t(0x0p+0f/*0.000000e+00*/);
  uint2 condval;
  if (((((((int)threadIdx.x) & 15) >> 3) + ((int)blockIdx.y)) < 4)) {
    condval = *(uint2*)(X + ((((((int)blockIdx.x) * 2048) + ((((int)threadIdx.x) >> 4) * 128)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) & 15) * 4)));
  } else {
    condval = make_uint2(__pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var));
  }
  *(uint2*)(((half_t*)X_shared) + (((int)threadIdx.x) * 4)) = condval;
  K_local[0] = K[(((int)threadIdx.x) & 31)];
  __syncthreads();
  for (int j = 0; j < 32; ++j) {
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
      float condval_1;
      if (((((j + (((int)threadIdx.x) & 31)) >> 5) + ((int)blockIdx.y)) < 4)) {
        condval_1 = (((float)((half_t*)X_shared)[((((i * 512) + ((((int)threadIdx.x) >> 5) * 64)) + j) + (((int)threadIdx.x) & 31))]) * ((float)K_local[0]));
      } else {
        condval_1 = 0x0p+0f/*0.000000e+00*/;
      }
      tmp[i] = condval_1;
    }
    #pragma unroll
    for (int i_1 = 0; i_1 < 2; ++i_1) {
      O_local_frag[i_1] = 0x0p+0f/*0.000000e+00*/;
      O_local_frag[i_1] = (O_local_frag[i_1] + tmp[i_1]);
      O_local_frag[i_1] = tl::AllReduce<tl::SumOp, 32, 1, 0, tl::NamedBarrier<256>>::run(O_local_frag[i_1]);
    }
    __syncthreads();
    if ((((int)threadIdx.x) % 32) == 0) {
      #pragma unroll
      for (int i_2 = 0; i_2 < 2; ++i_2) {
        ((float*)O_local)[((i_2 * 8) + (((int)threadIdx.x) >> 5))] = O_local_frag[i_2];
      }
    }
    __syncthreads();
    if (((int)threadIdx.x) < 16) {
      O[((((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 128)) + (((int)blockIdx.y) * 32)) + j)] = ((half_t)((float*)O_local)[((int)threadIdx.x)]);
    }
  }
}
