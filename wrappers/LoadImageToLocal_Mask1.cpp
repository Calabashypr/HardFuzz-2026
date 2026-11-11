
__aicore__ inline void LoadImageToLocal_Mask1(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        GM_ADDR src0Gm,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {
    (void)GenerateRandomMask(((data_length > 0) ? ((data_length > 64) ? 64 : data_length) : 1));
    (void)input0;
    AscendC::GlobalTensor<uint8_t> src0;
    src0.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(src0Gm), 1024U);
    AscendC::AippParams<half> cfg{};
    cfg.cPaddingParams.cPaddingMode = 0;
    cfg.cPaddingParams.cPaddingValue = (half)0;
    cfg.singleLineParams.isSingleLineCopy = false;
    AscendC::SetAippFunctions<uint8_t, half>(src0, AscendC::AippInputFormat::YUV400_U8, cfg);
    int32_t off = (offset_ptr + 31) & ~31; offset_ptr = off;
    AscendC::LocalTensor<half> dstHalf; dstHalf.SetAddrWithOffset(sharedTmpBuffer, offset_ptr);
    int32_t dstElems = 16; dstHalf.SetSize(dstElems * (int32_t)sizeof(half)); offset_ptr += dstHalf.GetSize();
    AscendC::LoadImageToLocal(dstHalf, {4, 4, 0, 0, 4, 0, 0, 0, 0});
    int32_t outN = (data_length > dstElems) ? dstElems : data_length;
    for (int32_t i = 0; i < outN; ++i) output.SetValue(i, (float)dstHalf.GetValue(i));
    for (int32_t i = outN; i < data_length; ++i) output.SetValue(i, 0.0f);
}
