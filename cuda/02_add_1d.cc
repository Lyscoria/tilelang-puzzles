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

extern "C" __global__ void tl_add_1d_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C);
extern "C" __global__ void __launch_bounds__(256, 1) tl_add_1d_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C) {
  uint2 __1;
    uint2 v_ = *(uint2*)(A + ((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 4)));
    uint2 v__1 = *(uint2*)(B + ((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 4)));
    *(uint1*)(&(__1.x)) = tl::to_uint1(tl::add2(tl::from_uint1<__half2>(*(uint1*)(&(v_.x))), tl::from_uint1<__half2>(*(uint1*)(&(v__1.x)))));
    *(uint1*)(&(__1.y)) = tl::to_uint1(tl::add2(tl::from_uint1<__half2>(*(uint1*)(&(v_.y))), tl::from_uint1<__half2>(*(uint1*)(&(v__1.y)))));
  *(uint2*)(C + ((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 4))) = __1;
}