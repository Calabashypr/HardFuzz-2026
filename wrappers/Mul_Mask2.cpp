
__aicore__ inline void Mul_Mask2(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const AscendC::LocalTensor<float> &input1,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {
    int32_t off = (offset_ptr + 31) & ~31;
    offset_ptr = off;
    AscendC::LocalTensor<float> b0 = sharedTmpBuffer.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];
    b0.SetSize(64 * (int32_t)sizeof(float));
    offset_ptr += b0.GetSize();
    off = (offset_ptr + 31) & ~31;
    offset_ptr = off;
    AscendC::LocalTensor<float> b1 = sharedTmpBuffer.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];
    b1.SetSize(64 * (int32_t)sizeof(float));
    offset_ptr += b1.GetSize();
    off = (offset_ptr + 31) & ~31;
    offset_ptr = off;
    AscendC::LocalTensor<float> bd = sharedTmpBuffer.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];
    bd.SetSize(64 * (int32_t)sizeof(float));
    offset_ptr += bd.GetSize();
    const AscendC::BinaryRepeatParams brp{1, 1, 1, 8, 8, 8};
    const int32_t len = (data_length > 0) ? data_length : 1;
    for (int32_t base = 0; base < len; base += 64) {
        int32_t chunk = len - base;
        if (chunk > 64) {
            chunk = 64;
        }
        for (int32_t i = 0; i < chunk; ++i) {
            b0.SetValue(i, input0.GetValue(base + i));
            b1.SetValue(i, input1.GetValue(base + i));
        }
        const uint64_t mask = (uint64_t)chunk;
        AscendC::Mul<float>(bd, b0, b1, mask, (uint8_t)1, brp);
        for (int32_t i = 0; i < chunk; ++i) {
            const_cast<AscendC::LocalTensor<float> &>(output).SetValue(base + i, bd.GetValue(i));
        }
    }
}
