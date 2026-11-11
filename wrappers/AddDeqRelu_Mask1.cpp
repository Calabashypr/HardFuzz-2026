
__aicore__ inline void AddDeqRelu_Mask1(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const AscendC::LocalTensor<float> &input1,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {
    int32_t off = (offset_ptr + 31) & ~31;
    offset_ptr = off;
    AscendC::LocalTensor<int32_t> b0 = sharedTmpBuffer.ReinterpretCast<int32_t>()[offset_ptr / (int32_t)sizeof(int32_t)];
    b0.SetSize(64 * (int32_t)sizeof(int32_t));
    offset_ptr += b0.GetSize();
    off = (offset_ptr + 31) & ~31;
    offset_ptr = off;
    AscendC::LocalTensor<int32_t> b1 = sharedTmpBuffer.ReinterpretCast<int32_t>()[offset_ptr / (int32_t)sizeof(int32_t)];
    b1.SetSize(64 * (int32_t)sizeof(int32_t));
    offset_ptr += b1.GetSize();
    off = (offset_ptr + 31) & ~31;
    offset_ptr = off;
    AscendC::LocalTensor<half> bd = sharedTmpBuffer.ReinterpretCast<half>()[offset_ptr / (int32_t)sizeof(half)];
    bd.SetSize(64 * (int32_t)sizeof(half));
    offset_ptr += bd.GetSize();
    const AscendC::BinaryRepeatParams brp{1, 1, 1, 8, 8, 8};
    half deq_scale = (half)0.1f;
    AscendC::SetDeqScale(deq_scale);
    const int32_t len = (data_length > 0) ? data_length : 1;
    for (int32_t base = 0; base < len; base += 64) {
        int32_t chunk = len - base;
        if (chunk > 64) {
            chunk = 64;
        }
        for (int32_t i = 0; i < chunk; ++i) {
            b0.SetValue(i, static_cast<int32_t>(input0.GetValue(base + i)));
            b1.SetValue(i, static_cast<int32_t>(input1.GetValue(base + i)));
        }
        uint64_t maskBits = GenerateRandomMask(chunk);
        uint64_t maskArr[2] = {maskBits, 0};
        for (int32_t zero_lane = 0; zero_lane < chunk; ++zero_lane) { bd.SetValue(zero_lane, static_cast<half>(0)); }
        AscendC::AddDeqRelu(bd, b0, b1, maskArr, (uint8_t)1, brp);
        for (int32_t i = 0; i < chunk; ++i) {
            const_cast<AscendC::LocalTensor<float> &>(output).SetValue(base + i, (float)bd.GetValue(i));
        }
    }
}
