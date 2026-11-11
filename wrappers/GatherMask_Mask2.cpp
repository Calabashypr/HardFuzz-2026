
__aicore__ inline void GatherMask_Mask2(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const AscendC::LocalTensor<float> &input1,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {
    (void)input1;
    (void)sharedTmpBuffer;
    (void)offset_ptr;
    const uint8_t src1Pattern = 7;
    uint32_t mask = (uint32_t)((data_length > 0) ? data_length : 1);
    uint64_t rsvdCnt = 0;
    AscendC::GatherMaskParams gmp{1, 1, 0, 0};
    AscendC::GatherMask<float>(const_cast<AscendC::LocalTensor<float> &>(output), input0, src1Pattern, true, mask, gmp,
            rsvdCnt);
    float v = output.GetValue(0);
    int32_t len = (data_length > 0) ? data_length : 1;
    for (int32_t i = 0; i < len; ++i) {
        const_cast<AscendC::LocalTensor<float> &>(output).SetValue(i, v);
    }
}
