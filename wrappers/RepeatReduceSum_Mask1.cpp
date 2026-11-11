
__aicore__ inline void RepeatReduceSum_Mask1(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {
    (void)GenerateRandomMask(((data_length > 0) ? ((data_length > 64) ? 64 : data_length) : 1));
    int32_t off = (offset_ptr + 31) & ~31;
    offset_ptr = off;
    AscendC::LocalTensor<float> buf = sharedTmpBuffer.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];
    buf.SetSize(64 * (int32_t)sizeof(float));
    offset_ptr += buf.GetSize();
    off = (offset_ptr + 31) & ~31;
    offset_ptr = off;
    AscendC::LocalTensor<float> wtmp = sharedTmpBuffer.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];
    wtmp.SetSize(64 * (int32_t)sizeof(float));
    offset_ptr += wtmp.GetSize();
    const int32_t len = (data_length > 0) ? data_length : 1;
    for (int32_t base = 0; base < len; base += 64) {
        int32_t chunk = len - base;
        if (chunk > 64) chunk = 64;
        for (int32_t i = 0; i < chunk; ++i) buf.SetValue(i, input0.GetValue(base + i));
        int32_t m = chunk;
        if (m < 1) m = 1;
        AscendC::RepeatReduceSum<float>(wtmp, buf, 1, m, 0, 1, 1, 8);
        float v = wtmp.GetValue(0);
        for (int32_t i = 0; i < chunk; ++i)
            const_cast<AscendC::LocalTensor<float> &>(output).SetValue(base + i, v);
    }
}
