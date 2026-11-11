
__aicore__ inline void Axpy_Mask1(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const AscendC::LocalTensor<float> &input1,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {
    int32_t off = (offset_ptr + 31) & ~31;
    offset_ptr = off;
    AscendC::LocalTensor<float> bx = sharedTmpBuffer.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];
    bx.SetSize(64 * (int32_t)sizeof(float));
    offset_ptr += bx.GetSize();
    off = (offset_ptr + 31) & ~31;
    offset_ptr = off;
    AscendC::LocalTensor<float> bd = sharedTmpBuffer.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];
    bd.SetSize(64 * (int32_t)sizeof(float));
    offset_ptr += bd.GetSize();
    const AscendC::UnaryRepeatParams urp{1, 1, 8, 8};
    const float alpha = 2.0f;
    const int32_t len = (data_length > 0) ? data_length : 1;
    for (int32_t base = 0; base < len; base += 64) {
        int32_t chunk = len - base;
        if (chunk > 64) {
            chunk = 64;
        }
        for (int32_t i = 0; i < chunk; ++i) {
            bx.SetValue(i, input0.GetValue(base + i));
            bd.SetValue(i, input1.GetValue(base + i));
        }
        uint64_t maskBits = GenerateRandomMask(chunk);
        uint64_t maskArr[2] = {maskBits, 0};
        for (int32_t zero_lane = 0; zero_lane < chunk; ++zero_lane) { bd.SetValue(zero_lane, static_cast<float>(0)); }
        AscendC::Axpy<float, float>(bd, bx, alpha, maskArr, (uint8_t)1, urp);
        for (int32_t i = 0; i < chunk; ++i) {
            const_cast<AscendC::LocalTensor<float> &>(output).SetValue(base + i, bd.GetValue(i));
        }
    }
}
