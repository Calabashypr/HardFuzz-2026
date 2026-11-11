
__aicore__ inline void AddReluCast_Mask2(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const AscendC::LocalTensor<float> &input1,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {
    int32_t off = (offset_ptr + 31) & ~31;
    offset_ptr = off;
    AscendC::LocalTensor<half> b0 = sharedTmpBuffer.ReinterpretCast<half>()[offset_ptr / (int32_t)sizeof(half)];
    b0.SetSize(64 * (int32_t)sizeof(half));
    offset_ptr += b0.GetSize();
    off = (offset_ptr + 31) & ~31;
    offset_ptr = off;
    AscendC::LocalTensor<half> b1 = sharedTmpBuffer.ReinterpretCast<half>()[offset_ptr / (int32_t)sizeof(half)];
    b1.SetSize(64 * (int32_t)sizeof(half));
    offset_ptr += b1.GetSize();
    off = (offset_ptr + 31) & ~31;
    offset_ptr = off;
    AscendC::LocalTensor<int8_t> bd8 = sharedTmpBuffer.ReinterpretCast<int8_t>()[offset_ptr / (int32_t)sizeof(int8_t)];
    bd8.SetSize(64 * (int32_t)sizeof(int8_t));
    offset_ptr += bd8.GetSize();
    const AscendC::BinaryRepeatParams brp{1, 1, 1, 8, 8, 8};
    const int32_t len = (data_length > 0) ? data_length : 1;
    for (int32_t base = 0; base < len; base += 64) {
        int32_t chunk = len - base;
        if (chunk > 64) chunk = 64;
        for (int32_t i = 0; i < chunk; ++i) {
            b0.SetValue(i, (half)input0.GetValue(base + i));
            b1.SetValue(i, (half)input1.GetValue(base + i));
        }
        const uint64_t mask = (uint64_t)chunk;
        AscendC::AddReluCast<int8_t, half>(bd8, b0, b1, mask, (uint8_t)1, brp);
        for (int32_t i = 0; i < chunk; ++i) {
            const int32_t iv = (int32_t)bd8.GetValue(i);
            const_cast<AscendC::LocalTensor<float> &>(output).SetValue(base + i, (float)iv);
        }
    }
}
