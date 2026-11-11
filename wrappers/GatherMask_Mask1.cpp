
__aicore__ inline void GatherMask_Mask1(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const AscendC::LocalTensor<float> &input1,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {
    (void)GenerateRandomMask(((data_length > 0) ? ((data_length > 64) ? 64 : data_length) : 1));
    (void)input1;
    int32_t off = (offset_ptr + 31) & ~31;
    offset_ptr = off;
    AscendC::LocalTensor<float> chunkDst = sharedTmpBuffer.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];
    chunkDst.SetSize(64 * (int32_t)sizeof(float));
    offset_ptr += chunkDst.GetSize();
    AscendC::LocalTensor<float> chunkSrc = sharedTmpBuffer.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];
    chunkSrc.SetSize(64 * (int32_t)sizeof(float));
    offset_ptr += chunkSrc.GetSize();
    const uint8_t src1Pattern = 7;
    const int32_t len = (data_length > 0) ? data_length : 1;
    AscendC::GatherMaskParams gmp{1, 1, 0, 0};
    for (int32_t base = 0; base < len; base += 64) {
        int32_t chunk = len - base;
        if (chunk > 64) {
            chunk = 64;
        }
        for (int32_t i = 0; i < chunk; ++i) {
            chunkSrc.SetValue(i, input0.GetValue(base + i));
        }
        uint64_t rsvdCnt = 0;
        AscendC::GatherMask<float>(chunkDst, chunkSrc, src1Pattern, true, (uint32_t)chunk, gmp, rsvdCnt);
        float v = chunkDst.GetValue(0);
        for (int32_t i = 0; i < chunk; ++i) {
            const_cast<AscendC::LocalTensor<float> &>(output).SetValue(base + i, v);
        }
    }
}
