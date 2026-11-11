
__aicore__ inline void ToBfloat16_Mask1(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {
    (void)sharedTmpBuffer;
    (void)offset_ptr;
    const int32_t len = (data_length > 0) ? data_length : 1;
    for (int32_t base = 0; base < len; base += 64) {
        int32_t chunk = len - base;
        if (chunk > 64) chunk = 64;
        uint64_t maskBits = GenerateRandomMask(chunk);
        for (int32_t i = 0; i < chunk; ++i) {
            if (((maskBits >> (uint32_t)i) & 1ULL) == 0ULL) {
                continue;
            }
            float v = input0.GetValue(base + i);
            bfloat16_t bf = AscendC::ToBfloat16(v);
            float vo = AscendC::ToFloat(bf);
            const_cast<AscendC::LocalTensor<float> &>(output).SetValue(base + i, vo);
        }
    }
}
