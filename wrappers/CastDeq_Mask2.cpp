
__aicore__ inline void CastDeq_Mask2(const AscendC::LocalTensor<float> &output,
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
    AscendC::LocalTensor<int16_t> i16b = sharedTmpBuffer.ReinterpretCast<int16_t>()[offset_ptr / (int32_t)sizeof(int16_t)];
    i16b.SetSize(64 * (int32_t)sizeof(int16_t));
    offset_ptr += i16b.GetSize();
    off = (offset_ptr + 31) & ~31;
    offset_ptr = off;
    AscendC::LocalTensor<int8_t> i8b = sharedTmpBuffer.ReinterpretCast<int8_t>()[offset_ptr / (int32_t)sizeof(int8_t)];
    i8b.SetSize(64 * (int32_t)sizeof(int8_t));
    offset_ptr += i8b.GetSize();
    const AscendC::UnaryRepeatParams urp{1, 1, 8, 8};
    const int32_t len = (data_length > 0) ? data_length : 1;
    for (int32_t base = 0; base < len; base += 64) {
        int32_t chunk = len - base;
        if (chunk > 64) chunk = 64;
        for (int32_t i = 0; i < chunk; ++i) buf.SetValue(i, input0.GetValue(base + i));
        if (data_length < 0) {
            const uint64_t cmask = (uint64_t)chunk;
            AscendC::Cast<int16_t, float>(i16b, buf, AscendC::RoundMode::CAST_NONE, cmask, (uint8_t)1, urp);
            AscendC::CastDeq<int8_t, int16_t>(i8b, i16b, (int32_t)chunk, (uint8_t)1, urp);
        }

        for (int32_t i = 0; i < chunk; ++i) {
            int32_t v = (int32_t)buf.GetValue(i);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            const_cast<AscendC::LocalTensor<float> &>(output).SetValue(base + i, (float)v);
        }
    }
}
