
__aicore__ inline void SetFmatrix_Mask2(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {
    int32_t rawI = (int32_t)input0.GetValue(0);
    uint16_t l1H = (uint16_t)(rawI > 1 ? (rawI < 32767 ? (uint16_t)rawI : 32767U) : 1U);
    uint16_t l1W = (uint16_t)((data_length > 0) ? (data_length < 32767 ? data_length : 32767) : 1);
    uint8_t padList[4] = {0, 0, 0, 0};
    AscendC::SetFmatrix(l1H, l1W, padList, AscendC::FmatrixMode::FMATRIX_LEFT);
    int32_t off = (offset_ptr + 31) & ~31;
    offset_ptr = off;
    AscendC::LocalTensor<float> srcb = sharedTmpBuffer.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];
    srcb.SetSize(64 * (int32_t)sizeof(float));
    offset_ptr += srcb.GetSize();
    off = (offset_ptr + 31) & ~31;
    offset_ptr = off;
    AscendC::LocalTensor<float> dstb = sharedTmpBuffer.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];
    dstb.SetSize(64 * (int32_t)sizeof(float));
    offset_ptr += dstb.GetSize();
    const AscendC::UnaryRepeatParams urp{1, 1, 8, 8};
    const int32_t len = (data_length > 0) ? data_length : 1;
    for (int32_t base = 0; base < len; base += 64) {
        int32_t chunk = len - base;
        if (chunk > 64) chunk = 64;
        for (int32_t i = 0; i < chunk; ++i) srcb.SetValue(i, input0.GetValue(base + i));
        uint64_t mask = (uint64_t)chunk;
        AscendC::Cast<float, float>(dstb, srcb, AscendC::RoundMode::CAST_RINT, mask, (uint8_t)1, urp);
        for (int32_t i = 0; i < chunk; ++i) {
            const_cast<AscendC::LocalTensor<float> &>(output).SetValue(base + i, dstb.GetValue(i));
        }
    }
}
