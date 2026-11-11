
__aicore__ inline void ScalarCast_Mask1(const AscendC::LocalTensor<float> &output,
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
            int32_t vi = AscendC::ScalarCast<float, int32_t, AscendC::RoundMode::CAST_ROUND>(v);
            float vo = AscendC::ScalarCast<int32_t, float, AscendC::RoundMode::CAST_NONE>(vi);
            const_cast<AscendC::LocalTensor<float> &>(output).SetValue(base + i, vo);
        }
    }
}
