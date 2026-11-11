
__aicore__ inline void Cast_Mask1(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {
    int32_t off = (offset_ptr + 31) & ~31;
    offset_ptr = off;
    AscendC::LocalTensor<float> buf = sharedTmpBuffer.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];
    buf.SetSize(64 * (int32_t)sizeof(float));
    offset_ptr += buf.GetSize();
    off = (offset_ptr + 31) & ~31;
    offset_ptr = off;
    AscendC::LocalTensor<int32_t> i32b = sharedTmpBuffer.ReinterpretCast<int32_t>()[offset_ptr / (int32_t)sizeof(int32_t)];
    i32b.SetSize(64 * (int32_t)sizeof(int32_t));
    offset_ptr += i32b.GetSize();
    off = (offset_ptr + 31) & ~31;
    offset_ptr = off;
    AscendC::LocalTensor<float> bd = sharedTmpBuffer.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];
    bd.SetSize(64 * (int32_t)sizeof(float));
    offset_ptr += bd.GetSize();
    const AscendC::UnaryRepeatParams urp{1, 1, 8, 8};
    const int32_t len = (data_length > 0) ? data_length : 1;
    for (int32_t base = 0; base < len; base += 64) {
        int32_t chunk = len - base;
        if (chunk > 64) chunk = 64;
        for (int32_t i = 0; i < chunk; ++i) {
            buf.SetValue(i, input0.GetValue(base + i));
        }

        uint64_t maskBits = GenerateRandomMask(chunk);
        uint64_t maskArr[2] = {maskBits, 0};
        for (int32_t zero_lane = 0; zero_lane < chunk; ++zero_lane) { i32b.SetValue(zero_lane, static_cast<int32_t>(0)); }
        AscendC::Cast<int32_t, float>(i32b, buf, AscendC::RoundMode::CAST_NONE, maskArr, (uint8_t)1, urp);
        for (int32_t zero_lane = 0; zero_lane < chunk; ++zero_lane) { bd.SetValue(zero_lane, static_cast<float>(0)); }
        AscendC::Cast<float, int32_t>(bd, i32b, AscendC::RoundMode::CAST_NONE, maskArr, (uint8_t)1, urp);
        for (int32_t i = 0; i < chunk; ++i) {
            const_cast<AscendC::LocalTensor<float> &>(output).SetValue(base + i, bd.GetValue(i));
        }
    }
}
