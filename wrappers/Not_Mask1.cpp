

__aicore__ inline void Not_Mask1(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {
    int32_t off = (offset_ptr + 31) & ~31;
    offset_ptr = off;
    AscendC::LocalTensor<int16_t> in16 = sharedTmpBuffer.ReinterpretCast<int16_t>()[offset_ptr / (int32_t)sizeof(int16_t)];
    in16.SetSize(64 * (int32_t)sizeof(int16_t));
    offset_ptr += in16.GetSize();
    off = (offset_ptr + 31) & ~31;
    offset_ptr = off;
    AscendC::LocalTensor<int16_t> out16 = sharedTmpBuffer.ReinterpretCast<int16_t>()[offset_ptr / (int32_t)sizeof(int16_t)];
    out16.SetSize(64 * (int32_t)sizeof(int16_t));
    offset_ptr += out16.GetSize();

    const int32_t len = (data_length > 0) ? data_length : 1;
    for (int32_t base = 0; base < len; base += 64) {
        int32_t chunk = len - base;
        if (chunk > 64) {
            chunk = 64;
        }
        for (int32_t i = 0; i < chunk; ++i) {
            in16.SetValue(i, static_cast<int16_t>(input0.GetValue(base + i)));
        }
        uint64_t m0 = GenerateRandomMask(chunk);
        uint64_t maskTwo[2] = {m0, 0ULL};
        for (int32_t zero_lane = 0; zero_lane < chunk; ++zero_lane) { out16.SetValue(zero_lane, static_cast<int16_t>(0)); }
        AscendC::Not<int16_t>(out16, in16, maskTwo, (uint8_t)1, {1, 1, 8, 8});
        for (int32_t i = 0; i < chunk; ++i) {
            const_cast<AscendC::LocalTensor<float> &>(output).SetValue(base + i, static_cast<float>(out16.GetValue(i)));
        }
    }
}
