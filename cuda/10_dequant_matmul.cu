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

extern "C" __global__ void tl_dequant_matmul_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, half_t* __restrict__ C);
extern "C" __global__ void __launch_bounds__(384, 1) tl_dequant_matmul_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, half_t* __restrict__ C) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* A_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* B_shared = ((void*)((char*)buf_dyn_shmem + 49152));
  void* B_deqt = ((void*)((char*)buf_dyn_shmem + 61440));
  __shared__ __align__(16) uint64_t mbarrier_mem[12];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  float C_local[64];
  half_t C_local_cast[2];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(A_desc);
    tl::prefetch_tma_descriptor(B_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    mbarrier[0].init(1);
    mbarrier[1].init(1);
    mbarrier[2].init(1);
    mbarrier[3].init(1);
    mbarrier[4].init(1);
    mbarrier[5].init(1);
    mbarrier[6].init(256);
    mbarrier[7].init(256);
    mbarrier[8].init(256);
    mbarrier[9].init(256);
    mbarrier[10].init(256);
    mbarrier[11].init(256);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (256 <= ((int)threadIdx.x)) {
    tl::warpgroup_reg_dealloc<24>();
    for (int idx_k = 0; idx_k < 64; ++idx_k) {
      mbarrier[((idx_k % 3) + 6)].wait((((idx_k % 6) / 3) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[(idx_k % 3)].arrive_and_expect_tx(16384);
        tl::tma_load(A_desc, mbarrier[(idx_k % 3)], (&(((half_t*)A_shared)[((idx_k % 3) * 8192)])), (idx_k * 64), (((int)blockIdx.x) * 128));
      }
      mbarrier[((idx_k % 3) + 9)].wait((((idx_k % 6) / 3) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[((idx_k % 3) + 3)].arrive_and_expect_tx(4096);
        tl::tma_load(B_desc, mbarrier[((idx_k % 3) + 3)], (&(((uchar*)B_shared)[((idx_k % 3) * 4096)])), (((int)blockIdx.y) * 64), (idx_k * 64));
      }
    }
  } else {
    tl::warpgroup_reg_alloc<240>();
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(C_local + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    for (int idx_k_1 = 0; idx_k_1 < 64; ++idx_k_1) {
      mbarrier[((idx_k_1 % 3) + 3)].wait(((idx_k_1 % 6) / 3));
      tl::__sync_thread_partial(3, 256);
      #pragma unroll
      for (int i_1 = 0; i_1 < 16; ++i_1) {
        ((half_t*)B_deqt)[(((((((((((int)threadIdx.x) & 63) >> 5) * 4096) + (i_1 * 256)) + ((((int)threadIdx.x) >> 6) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_1 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))] = ((half_t)(((float)((half_t)(((uchar*)B_shared)[((((idx_k_1 % 3) * 4096) + (i_1 * 256)) + ((int)threadIdx.x))] & (uchar)15))) - 0x1p+3f/*8.000000e+00*/));
        ((half_t*)B_deqt)[((((((((((((int)threadIdx.x) & 63) >> 5) * 4096) + (i_1 * 256)) + ((((int)threadIdx.x) >> 6) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_1 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1)] = ((half_t)(((float)((half_t)((((uchar*)B_shared)[((((idx_k_1 % 3) * 4096) + (i_1 * 256)) + ((int)threadIdx.x))] >> (uchar)4) & (uchar)15))) - 0x1p+3f/*8.000000e+00*/));
      }
      mbarrier[((idx_k_1 % 3) + 9)].arrive();
      mbarrier[(idx_k_1 % 3)].wait(((idx_k_1 % 6) / 3));
      {
        tl::GmmaDescriptor desc_a;
        tl::GmmaDescriptor desc_b;
        tl::__sync_thread_partial(3, 256);
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, (&(((half_t*)A_shared)[((idx_k_1 % 3) * 8192)])));
        tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b, (&(((half_t*)B_deqt)[0])));
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(C_local + 0), 64);
        tl::warpgroup_arrive();
        tl::fence_proxy_async();
        #pragma unroll
        for (int i_2 = 0; i_2 < 2; ++i_2) {
          #pragma unroll
          for (int ki = 0; ki < 4; ++ki) {
            tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(uint64_t(desc_a + (((i_2 * 8192) + (ki * 32)) >> 4)), uint64_t(desc_b + ((((((int)threadIdx.x) >> 7) * 8192) + (ki * 2048)) >> 4)), ((uint32_t*)(C_local + (i_2 * 32))), 1);
          }
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(C_local + 0), 64);
      }
      mbarrier[((idx_k_1 % 3) + 6)].arrive();
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 32; ++i_3) {
      uint1 __1;
      float2 v_ = *(float2*)(C_local + (i_3 * 2));
      ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
      *(uint1*)(C_local_cast + 0) = __1;
      *(uint1*)(C + (((((((((((int)blockIdx.x) * 524288) + ((i_3 >> 4) * 262144)) + (((((int)threadIdx.x) & 127) >> 5) * 65536)) + ((i_3 & 1) * 32768)) + (((((int)threadIdx.x) & 31) >> 2) * 4096)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) >> 7) * 64)) + (((i_3 & 15) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(C_local_cast + 0);
    }
  }
}