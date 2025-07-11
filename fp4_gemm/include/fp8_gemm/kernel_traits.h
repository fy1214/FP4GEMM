#pragma once

#include "cute/algorithm/copy.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include <cutlass/numeric_types.h>

using namespace cute;

template<
    int kBlockK_=128, 
    int kBlockM_=128, 
    int kBlockN_=128, 
    int kNWarps_=4, 
    typename A_type=cutlass::float_e4m3_t,
    typename B_type=cutlass::float_e4m3_t,
    typename C_type=float
>
struct Flash_kernel_traits {
    using Type = A_type;
    using ElementAccum = float;
    using index_t = uint32_t;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 32;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kBlockK = kBlockK_;

    using MMA_Atom_Arch = MMA_Atom<
        SM120_16x8x32_TN<A_type, B_type, C_type>
    >;

    using ValLayoutMNK = Layout<Shape<_2, _8, _16>>;

    using TiledMma = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        Tile<Int<32 * kNWarps>, _128, _128>>;

    using GmemCopyAtom = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<sizeof(uint128_t)>, T>; 
    static constexpr int GmemValsPerLoad = sizeof(uint128_t) / sizeof(A_type); // 每次vector加载128位，可以加载几个元素。128/8=16
    static constexpr int GmemThreadsPerRow = kHeadDim / GmemValsPerLoad; // each thread reads 128 bit，计算得到需要几个线程  128/16=8  
    // 128 / 8 = 16
    // 128 / 16 = 8
    using TiledCopyABC = decltype(make_tiled_copy( // https://zhuanlan.zhihu.com/p/703560147
        GmemCopyAtom{},   // MMA_Atom
        make_layout(      // ThrLayout
            Shape<Int<kNThreads / GmemThreadsPerRow>, Int<GmemThreadsPerRow>>{},
            GenRowMajor{}),
        make_layout(Shape<_1, Int<GmemValsPerLoad>>{}, GenRowMajor{}))
    );  // ValLa
};