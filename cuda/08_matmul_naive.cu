#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/mma.h>
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void tl_matmul_naive_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C);
extern "C" __global__ void __launch_bounds__(128, 1) tl_matmul_naive_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C) {
  float C_local[128];
  half_t A_local[128];
  half_t B_local[128];
  half_t C_local_cast[2];
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(C_local + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  for (int bk = 0; bk < 64; ++bk) {
    #pragma unroll
    for (int i_1 = 0; i_1 < 64; ++i_1) {
      *(uint1*)(A_local + (i_1 * 2)) = *(uint1*)(A + (((((((((((int)blockIdx.x) * 524288) + (((((int)threadIdx.x) & 63) >> 5) * 262144)) + (((i_1 & 15) >> 2) * 65536)) + ((i_1 & 1) * 32768)) + (((((int)threadIdx.x) & 31) >> 2) * 4096)) + (bk * 64)) + ((i_1 >> 4) * 16)) + (((i_1 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 128; ++i_2) {
      B_local[i_2] = B[(((((((((bk * 262144) + ((i_2 >> 5) * 65536)) + (((i_2 & 3) >> 1) * 32768)) + ((((int)threadIdx.x) & 3) * 8192)) + ((i_2 & 1) * 4096)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) >> 6) * 64)) + (((i_2 & 31) >> 2) * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
    }
    {
      for (int ki = 0; ki < 4; ++ki) {
        for (int i_3 = 0; i_3 < 4; ++i_3) {
          for (int j = 0; j < 4; ++j) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + ((i_3 * 32) + (j * 8))), reinterpret_cast<const unsigned*>(A_local + ((ki * 32) + (i_3 * 8))), reinterpret_cast<const unsigned*>(B_local + ((ki * 32) + (j * 8))));
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + (((i_3 * 32) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + ((ki * 32) + (i_3 * 8))), reinterpret_cast<const unsigned*>(B_local + (((ki * 32) + (j * 8)) + 4)));
          }
        }
      }
    }
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 64; ++i_4) {
    uint1 __1;
    float2 v_ = *(float2*)(C_local + (i_4 * 2));
    ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
    *(uint1*)(C_local_cast + 0) = __1;
    *(uint1*)(C + (((((((((((int)blockIdx.x) * 524288) + (((((int)threadIdx.x) & 63) >> 5) * 262144)) + ((i_4 >> 4) * 65536)) + ((i_4 & 1) * 32768)) + (((((int)threadIdx.x) & 31) >> 2) * 4096)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) >> 6) * 64)) + (((i_4 & 15) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(C_local_cast + 0);
  }
}