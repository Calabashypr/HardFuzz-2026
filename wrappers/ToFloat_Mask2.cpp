
__aicore__ inline void ToFloat_Mask2(const AscendC::LocalTensor<float> &output,
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
        const uint64_t mask = (uint64_t)chunk;
        for (int32_t i = 0; i < (int32_t)mask; ++i) {
            float v = input0.GetValue(base + i);
            bfloat16_t bf = AscendC::ToBfloat16(v);
            float vo = AscendC::ToFloat(bf);
            const_cast<AscendC::LocalTensor<float> &>(output).SetValue(base + i, vo);
        }
    }
}
