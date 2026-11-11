
__aicore__ inline void Select_Mask2(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const AscendC::LocalTensor<float> &input1,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {
    int32_t off = (offset_ptr + 31) & ~31;
    offset_ptr = off;
    AscendC::LocalTensor<uint8_t> sel_bits = sharedTmpBuffer[offset_ptr];
    sel_bits.SetSize(32);
    offset_ptr += sel_bits.GetSize();
    off = (offset_ptr + 31) & ~31;
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
    AscendC::LocalTensor<float> bout = sharedTmpBuffer.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];
    bout.SetSize(64 * (int32_t)sizeof(float));
    offset_ptr += bout.GetSize();
    const AscendC::BinaryRepeatParams brp{1, 1, 1, 8, 8, 8};
    const int32_t len = (data_length > 0) ? data_length : 1;
    for (int32_t base = 0; base < len; base += 64) {
        int32_t chunk = len - base;
        if (chunk > 64) chunk = 64;
        for (int32_t i = 0; i < chunk; ++i) {
            b0.SetValue(i, input0.GetValue(base + i));
            b1.SetValue(i, input1.GetValue(base + i));
        }

        AscendC::Compare<float, uint8_t>(sel_bits, b0, b1, AscendC::CMPMODE::LT, (uint64_t)chunk, (uint8_t)1, brp);
        AscendC::Select<float, uint8_t>(bout, sel_bits, b0, b1, AscendC::SELMODE::VSEL_CMPMASK_SPR, (uint64_t)chunk, (uint8_t)1, brp);
        for (int32_t i = 0; i < chunk; ++i) {
            const_cast<AscendC::LocalTensor<float> &>(output).SetValue(base + i, bout.GetValue(i));
        }
    }
}
