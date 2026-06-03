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

extern "C" __global__ void tl_matmul_opt_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, half_t* __restrict__ C);
extern "C" __global__ void __launch_bounds__(256, 1) tl_matmul_opt_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, half_t* __restrict__ C) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* A_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* B_shared = ((void*)((char*)buf_dyn_shmem + 49152));
  __shared__ __align__(16) uint64_t mbarrier_mem[6];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  float C_local[128];
  half_t C_local_cast[2];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(A_desc);
    tl::prefetch_tma_descriptor(B_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    mbarrier[0].init(1);
    mbarrier[1].init(1);
    mbarrier[2].init(1);
    mbarrier[3].init(128);
    mbarrier[4].init(128);
    mbarrier[5].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (128 <= ((int)threadIdx.x)) {
    tl::warpgroup_reg_dealloc<24>();
    for (int bk = 0; bk < 64; ++bk) {
      mbarrier[((bk % 3) + 3)].wait((((bk % 6) / 3) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[(bk % 3)].expect_transaction(16384);
        tl::tma_load(A_desc, mbarrier[(bk % 3)], (&(((half_t*)A_shared)[((bk % 3) * 8192)])), (bk * 64), (((int)blockIdx.x) * 128));
        mbarrier[(bk % 3)].arrive_and_expect_tx(16384);
        tl::tma_load(B_desc, mbarrier[(bk % 3)], (&(((half_t*)B_shared)[((bk % 3) * 8192)])), (((int)blockIdx.y) * 128), (bk * 64));
        tl::tma_load(B_desc, mbarrier[(bk % 3)], (&(((half_t*)B_shared)[(((bk % 3) * 8192) + 4096)])), ((((int)blockIdx.y) * 128) + 64), (bk * 64));
      }
    }
  } else {
    tl::warpgroup_reg_alloc<240>();
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(C_local + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    for (int bk_1 = 0; bk_1 < 64; ++bk_1) {
      mbarrier[(bk_1 % 3)].wait(((bk_1 % 6) / 3));
      {
        tl::GmmaDescriptor desc_a;
        tl::GmmaDescriptor desc_b;
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, (&(((half_t*)A_shared)[((bk_1 % 3) * 8192)])));
        tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b, (&(((half_t*)B_shared)[((bk_1 % 3) * 8192)])));
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(C_local + 0), 128);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int i_1 = 0; i_1 < 2; ++i_1) {
          #pragma unroll
          for (int ki = 0; ki < 4; ++ki) {
            tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(uint64_t(desc_a + (((i_1 * 8192) + (ki * 32)) >> 4)), uint64_t(desc_b + ((ki * 2048) >> 4)), ((uint32_t*)(C_local + (i_1 * 64))), 1);
          }
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(C_local + 0), 128);
      }
      mbarrier[((bk_1 % 3) + 3)].arrive();
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 64; ++i_2) {
      uint1 __1;
      float2 v_ = *(float2*)(C_local + (i_2 * 2));
      ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
      *(uint1*)(C_local_cast + 0) = __1;
      *(uint1*)(C + ((((((((((int)blockIdx.x) * 524288) + ((i_2 >> 5) * 262144)) + ((((int)threadIdx.x) >> 5) * 65536)) + ((i_2 & 1) * 32768)) + (((((int)threadIdx.x) & 31) >> 2) * 4096)) + (((int)blockIdx.y) * 128)) + (((i_2 & 31) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(C_local_cast + 0);
    }
  }
}