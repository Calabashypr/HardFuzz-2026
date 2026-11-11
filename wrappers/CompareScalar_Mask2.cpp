
__aicore__ inline void CompareScalar_Mask2(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const AscendC::LocalTensor<float> &input1,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {
    int32_t off = (offset_ptr + 31) & ~31;
    offset_ptr = off;
    AscendC::LocalTensor<uint8_t> bitmask = sharedTmpBuffer[offset_ptr];
    bitmask.SetSize(32);
    offset_ptr += bitmask.GetSize();
    off = (offset_ptr + 31) & ~31;
    offset_ptr = off;
    AscendC::LocalTensor<float> buf = sharedTmpBuffer.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];
    buf.SetSize(64 * (int32_t)sizeof(float));
    offset_ptr += buf.GetSize();
    const AscendC::UnaryRepeatParams urp{1, 1, 8, 8};
    const int32_t len = (data_length > 0) ? data_length : 1;
    const float scalar_val = (len > 0) ? input1.GetValue(0) : 0.0f;
    for (int32_t base = 0; base < len; base += 64) {
        int32_t chunk = len - base;
        if (chunk > 64) {
            chunk = 64;
        }
        for (int32_t i = 0; i < chunk; ++i) {
            buf.SetValue(i, input0.GetValue(base + i));
        }
        const uint64_t mask = (uint64_t)chunk;
        AscendC::CompareScalar<float, uint8_t>(
                bitmask, buf, scalar_val, AscendC::CMPMODE::LT, mask, (uint8_t)1, urp);
        for (int32_t i = 0; i < chunk; ++i) {
            uint8_t byte_val = bitmask.GetValue(i >> 3);
            uint8_t bit = (byte_val >> (uint8_t)(i & 7)) & (uint8_t)0x1;
            const_cast<AscendC::LocalTensor<float> &>(output).SetValue(base + i, (bit != 0U) ? 1.0f : 0.0f);
        }
    }
}
