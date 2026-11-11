
__aicore__ inline void ScalarGetCountOfValue_Mask2(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {
    int32_t off = (offset_ptr + 31) & ~31;
    offset_ptr = off;
    AscendC::LocalTensor<float> buf = sharedTmpBuffer.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];
    buf.SetSize(64 * (int32_t)sizeof(float));
    offset_ptr += buf.GetSize();
    AscendC::LocalTensor<float> bd = sharedTmpBuffer.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];
    bd.SetSize(64 * (int32_t)sizeof(float));
    offset_ptr += bd.GetSize();
    const AscendC::UnaryRepeatParams urp{1, 1, 8, 8};
    const int32_t len = (data_length > 0) ? data_length : 1;
    for (int32_t base = 0; base < len; base += 64) {
        int32_t chunk = len - base;
        if (chunk > 64) {
            chunk = 64;
        }
        for (int32_t i = 0; i < chunk; ++i) {
            buf.SetValue(i, input0.GetValue(base + i));
        }
        const uint64_t mask = (uint64_t)chunk;
        AscendC::Cast<float, float>(bd, buf, AscendC::RoundMode::CAST_RINT, mask, (uint8_t)1, urp);
        for (int32_t i = 0; i < chunk; ++i) {
            union {
                float f;
                uint32_t u;
            } vu;
            vu.f = bd.GetValue(i);
            uint64_t word = static_cast<uint64_t>(vu.u);
            int64_t c = AscendC::ScalarGetCountOfValue<1>(word);
            const_cast<AscendC::LocalTensor<float> &>(output).SetValue(base + i, static_cast<float>(c));
        }
    }
}
