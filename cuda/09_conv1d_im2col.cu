#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/wgmma.h>
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void tl_conv1d_im2col_kernel(const half_t* __restrict__ K, half_t* __restrict__ O, const half_t* __restrict__ X);
extern "C" __global__ void __launch_bounds__(256, 1) tl_conv1d_im2col_kernel(const half_t* __restrict__ K, half_t* __restrict__ O, const half_t* __restrict__ X) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* X_reshaped = ((void*)((char*)buf_dyn_shmem + 0));
  void* K_shared = ((void*)((char*)buf_dyn_shmem + 32768));
  float O_reshaped[64];
  half_t O_local_cast[2];
  #pragma unroll
  for (int i = 0; i < 64; ++i) {
    half_t condval;
    if ((((((((int)blockIdx.y) * 32) + ((((int)threadIdx.x) & 3) * 8)) + ((((int)threadIdx.x) & 127) >> 2)) + (i & 7)) < 128)) {
      condval = X[(((((((((int)blockIdx.x) * 2048) + ((i >> 3) * 256)) + ((((int)threadIdx.x) >> 7) * 128)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) & 3) * 8)) + ((((int)threadIdx.x) & 127) >> 2)) + (i & 7))];
    } else {
      condval = half_t(0x0p+0f/*0.000000e+00*/);
    }
    ((half_t*)X_reshaped)[((((((i >> 3) * 2048) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + (i & 7))] = condval;
  }
  *(uint2*)(((half_t*)K_shared) + ((((((((int)threadIdx.x) & 7) >> 2) * 512) + ((((int)threadIdx.x) >> 3) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 1) * 4))) = *(uint2*)(K + (((int)threadIdx.x) * 4));
  {
    tl::GmmaDescriptor desc_a;
    tl::GmmaDescriptor desc_b;
    __syncthreads();
    tl::initialize_wgmma_descriptor<2, 1, 32>(desc_a, (&(((half_t*)X_reshaped)[0])));
    tl::initialize_wgmma_descriptor<3, 256, 16>(desc_b, (&(((half_t*)K_shared)[0])));
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(O_reshaped + 0), 64);
    tl::warpgroup_arrive();
    tl::fence_proxy_async();
    #pragma unroll
    for (int i_1 = 0; i_1 < 8; ++i_1) {
      #pragma unroll
      for (int ki = 0; ki < 2; ++ki) {
        tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 16, 16, false, true, 1, 1>(uint64_t(desc_a + (((i_1 * 4096) + (ki * 32)) >> 4)), uint64_t(desc_b + ((((((int)threadIdx.x) >> 7) * 1024) + (ki * 512)) >> 4)), ((uint32_t*)(O_reshaped + (i_1 * 8))), ((0 < ki) ? 1 : 0));
      }
    }
    tl::warpgroup_commit_batch();
    tl::warpgroup_wait<0>();
    tl::warpgroup_fence_operand(reinterpret_cast<float*>(O_reshaped + 0), 64);
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 32; ++i_2) {
    uint1 __1;
    float2 v_ = *(float2*)(O_reshaped + (i_2 * 2));
    ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
    *(uint1*)(O_local_cast + 0) = __1;
    *(uint1*)(O + ((((((((((((int)blockIdx.x) * 65536) + ((i_2 >> 2) * 8192)) + (((((int)threadIdx.x) & 127) >> 6) * 4096)) + (((int)blockIdx.y) * 1024)) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + ((i_2 & 1) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + ((((int)threadIdx.x) >> 7) * 16)) + (((i_2 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(O_local_cast + 0);
  }
}