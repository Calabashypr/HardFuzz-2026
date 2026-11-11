__aicore__ inline void Mean(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {

    constexpr uint32_t OUTTER = 1;
    constexpr uint32_t N = 8;
    constexpr uint32_t INNER = 8;
    int32_t off = (offset_ptr + 31) & ~31;
    offset_ptr = off;
    auto bufF = sharedTmpBuffer.ReinterpretCast<float>();
    AscendC::LocalTensor<float> src = bufF[off/4]; src.SetSize(OUTTER * INNER); off += (int32_t)(OUTTER * INNER * sizeof(float));
    AscendC::LocalTensor<float> dst = bufF[off/4]; dst.SetSize(8); off += (int32_t)(8 * sizeof(float));
    AscendC::LocalTensor<uint8_t> apiTmp; apiTmp.SetAddrWithOffset(sharedTmpBuffer, off); apiTmp.SetSize(1024); off += 1024; offset_ptr = off;
    for (int32_t i = 0; i < (int32_t)(OUTTER * INNER); ++i) src.SetValue(i, (data_length > 0) ? input0.GetValue(i % data_length) : 0.0f);
    AscendC::MeanParams p{OUTTER, INNER, N};
    AscendC::Mean<float, float, false, false, -1>(dst, src, apiTmp, p);
    float mv = dst.GetValue(0);
    for (int32_t i = 0; i < data_length; ++i) output.SetValue(i, mv);
}
