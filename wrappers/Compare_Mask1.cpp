
__aicore__ inline void Compare_Mask1(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const AscendC::LocalTensor<float> &input1,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {
    int32_t off = (offset_ptr + 31) & ~31;
    offset_ptr = off;
    AscendC::LocalTensor<uint8_t> dst_bits = sharedTmpBuffer[offset_ptr];
    dst_bits.SetSize(32);
    offset_ptr += dst_bits.GetSize();
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
        uint64_t maskBits = GenerateRandomMask(chunk);
        uint64_t maskArr[2] = {maskBits, 0ULL};
        AscendC::Compare<float, uint8_t>(
                dst_bits, b0, b1, AscendC::CMPMODE::LT, maskArr, (uint8_t)1, brp);
        for (int32_t i = 0; i < chunk; ++i) {
            uint8_t b = dst_bits.GetValue(i >> 3);
            uint8_t bit = (b >> (i & 7)) & (uint8_t)0x1;
            const_cast<AscendC::LocalTensor<float> &>(output).SetValue(base + i, (bit != 0U) ? 1.0f : 0.0f);
        }
    }
}
