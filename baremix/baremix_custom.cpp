#define ASCENDC_CUBE_ONLY
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "../ops/mmad.h"
using namespace matmul;

__aicore__ inline uint32_t Ceiling(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

__aicore__ inline void CopyTiling(TCubeTiling *tiling, GM_ADDR tilingGM)
{
    uint32_t *ptr = reinterpret_cast<uint32_t *>(tiling);
    auto tiling32 = reinterpret_cast<__gm__ uint32_t *>(tilingGM);

    for (uint32_t i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); i++, ptr++) {
        *ptr = *(tiling32 + i);
    }
    return;
}

template <typename cType> class LeakyReluKernel {
public:
    __aicore__ inline LeakyReluKernel(){};
    AscendC::GlobalTensor<cType> cGlobal;
    TCubeTiling tiling;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> reluInQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> reluOutQueue_;
};

// HF_PRE_COMPUTE

extern "C" __global__ __aicore__ void baremix_custom(GM_ADDR input1, GM_ADDR input2, GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c,
                                                              GM_ADDR workspace, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
#ifdef ASCENDC_CPU_DEBUG
    if ASCEND_IS_AIV { if (AscendC::GetSubBlockIdx() != 0) { return; } }
#endif
    AscendC::TPipe pipe;
    TCubeTiling tiling;
    CopyTiling(&tiling, tilingGm);

    int offset_ptr = 0;
    int tensor_len = 64;
    int active_len = 64;
    int tensor_len_1 = tiling.singleCoreM * tiling.singleCoreN;
    int tensor_len_a = tiling.M * tiling.Ka;

    AscendC::LocalTensor<uint8_t> tmp;
    bool ans = AscendC::PopStackBuffer<uint8_t, AscendC::TPosition::LCM>(tmp);
    (void)ans;

    // HF_TENSOR_ALLOCATIONS

    AscendC::LocalTensor<float> OutputLocal;
    OutputLocal.SetAddrWithOffset(tmp, offset_ptr);
    OutputLocal.SetSize(tensor_len_1 * sizeof(float));
    offset_ptr += OutputLocal.GetSize();

    AscendC::GlobalTensor<float> OutputGlobal;

    TCubeTiling matmulTiling = tiling;
    AscendC::GlobalTensor<float> input1Global;
    AscendC::GlobalTensor<int32_t> input2Global;
    AscendC::GlobalTensor<float> aGlobal;
    AscendC::GlobalTensor<float> bGlobal;
    AscendC::GlobalTensor<float> cGlobal;
    AscendC::GlobalTensor<float> biasGlobal;

    input1Global.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(input1), tensor_len);
    input2Global.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(input2), tensor_len);
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a), matmulTiling.M * matmulTiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(b), matmulTiling.Kb * matmulTiling.N);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(c), matmulTiling.M * matmulTiling.N);
    biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(bias), matmulTiling.N);

    if ASCEND_IS_AIC {

        int32_t blockIdx = AscendC::GetBlockIdx();
        auto mSingleBlocks = Ceiling(matmulTiling.M, matmulTiling.singleCoreM);
        auto mCoreIndx = blockIdx % mSingleBlocks;
        auto nCoreIndx = blockIdx / mSingleBlocks;
        int32_t offsetA = mCoreIndx * matmulTiling.Ka * matmulTiling.singleCoreM;
        int32_t offsetB = nCoreIndx * matmulTiling.singleCoreN;
        int32_t offsetC = mCoreIndx * matmulTiling.N * matmulTiling.singleCoreM + nCoreIndx * matmulTiling.singleCoreN;
        int32_t offsetBias = nCoreIndx * matmulTiling.singleCoreN;

        aGlobal = aGlobal[offsetA];
        bGlobal = bGlobal[offsetB];
        cGlobal = cGlobal[offsetC];
        biasGlobal = biasGlobal[offsetBias];

        Matmul<
            MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>,
            MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>,
            MatmulType<AscendC::TPosition::VECIN, CubeFormat::ND, float>,
            MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>
        > matmulObj;

        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulObj, &matmulTiling);
        matmulObj.SetTensorA(aGlobal);
        matmulObj.SetTensorB(bGlobal);
        matmulObj.SetBias(biasGlobal);
        matmulObj.IterateAll(cGlobal);
        matmulObj.End();

    }
    if ASCEND_IS_AIV {
        AscendC::LocalTensor<float> input1;
        input1.SetAddrWithOffset(tmp, offset_ptr);
        input1.SetSize(tensor_len * sizeof(float));
        offset_ptr += input1.GetSize();
        AscendC::Duplicate<float>(input1, (float)0.0, tensor_len);
        AscendC::DataCopy(input1, input1Global, tensor_len);

        AscendC::LocalTensor<int32_t> input2Local;
        input2Local.SetAddrWithOffset(tmp, offset_ptr);
        input2Local.SetSize(tensor_len * sizeof(int32_t));
        offset_ptr += input2Local.GetSize();
        AscendC::Duplicate<int32_t>(input2Local, (int)0, tensor_len);
        AscendC::DataCopy(input2Local, input2Global, tensor_len);

        // HF_COMPUTE_BODY

        offset_ptr += tensor_len_1 * sizeof(float);
        OutputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(c) + AscendC::GetBlockIdx() * tiling.M * tiling.N);
        AscendC::DataCopy(OutputLocal, cGlobal, tiling.singleCoreM * tiling.singleCoreN);

        // HF_COMPUTE_OUTPUT

    }
}
