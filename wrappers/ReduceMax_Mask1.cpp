
__aicore__ inline void ReduceMax_Mask1(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t sharedTmpBytes,
        const int32_t data_length) {
    constexpr int32_t kRf = 512;
    int32_t need = (int32_t)(64 * sizeof(float)) * 3 + (int32_t)(kRf * sizeof(float)) + 64;
    if (sharedTmpBytes - offset_ptr < need || data_length <= 0) return;
    int32_t off = (offset_ptr + 31) & ~31;
    offset_ptr = off;
    AscendC::LocalTensor<float> buf = sharedTmpBuffer.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];
    buf.SetSize(64 * (int32_t)sizeof(float));
    offset_ptr += buf.GetSize();
    off = (offset_ptr + 31) & ~31;
    offset_ptr = off;
    AscendC::LocalTensor<float> dstRed = sharedTmpBuffer.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];
    dstRed.SetSize(1 * (int32_t)sizeof(float));
    offset_ptr += dstRed.GetSize();
    off = (offset_ptr + 31) & ~31;
    offset_ptr = off;
    AscendC::LocalTensor<float> work = sharedTmpBuffer.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];
    work.SetSize(kRf * (int32_t)sizeof(float));
    offset_ptr += work.GetSize();
    const int32_t len = (data_length > 0) ? data_length : 1;
    for (int32_t base = 0; base < len; base += 64) {
        int32_t chunk = len - base;
        if (chunk > 64) chunk = 64;
        for (int32_t i = 0; i < chunk; ++i) buf.SetValue(i, input0.GetValue(base + i));
        uint64_t maskBits = GenerateRandomMask(chunk);
        uint64_t m2[2] = {maskBits, 0ULL};
        for (int32_t zero_lane = 0; zero_lane < chunk; ++zero_lane) { dstRed.SetValue(zero_lane, static_cast<float>(0)); }
        AscendC::ReduceMax<float>(dstRed, buf, work, m2, 1, 8, false);
        float v = dstRed.GetValue(0);
        for (int32_t i = 0; i < chunk; ++i)
            const_cast<AscendC::LocalTensor<float> &>(output).SetValue(base + i, v);
    }
}
