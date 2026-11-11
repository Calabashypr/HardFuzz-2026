__aicore__ inline void LoadImageToLocal(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        GM_ADDR src0Gm,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {

    (void)input0;

    AscendC::GlobalTensor<uint8_t> src0;
    src0.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(src0Gm), 1024U);
    AscendC::AippParams<half> aippConfig{};
    aippConfig.cPaddingParams.cPaddingMode = 0;
    aippConfig.cPaddingParams.cPaddingValue = (half)0;
    aippConfig.singleLineParams.isSingleLineCopy = false;
    AscendC::SetAippFunctions<uint8_t, half>(src0, AscendC::AippInputFormat::YUV400_U8, aippConfig);

    uint16_t horizSize = 4;
    uint16_t vertSize = (uint16_t)((data_length > 0) ? (data_length / horizSize) : 1);
    if (vertSize < 2) vertSize = 2;
    uint16_t horizStartPos = 0;
    uint16_t vertStartPos = 0;
    uint16_t srcHorizSize = horizSize;
    uint8_t topPadSize = 0;
    uint8_t botPadSize = 0;
    uint8_t leftPadSize = 0;
    uint8_t rightPadSize = 0;
    int32_t dstElems = (int32_t)horizSize * (int32_t)vertSize;
    int32_t dstOff = (offset_ptr + 31) & ~31;
    offset_ptr = dstOff;
    AscendC::LocalTensor<half> dstHalf;
    dstHalf.SetAddrWithOffset(sharedTmpBuffer, offset_ptr);
    dstHalf.SetSize(dstElems * (int32_t)sizeof(half));
    offset_ptr += dstHalf.GetSize();

    AscendC::LoadImageToLocal(dstHalf, { horizSize, vertSize, horizStartPos, vertStartPos, srcHorizSize, topPadSize, botPadSize, leftPadSize, rightPadSize });

    int32_t outElems = data_length;
    if (outElems < 0) outElems = 0;
    if (outElems > dstElems) outElems = dstElems;
    for (int32_t i = 0; i < outElems; ++i) { output.SetValue(i, (float)dstHalf.GetValue(i)); }

    for (int32_t i = outElems; i < data_length; ++i) { output.SetValue(i, 0.0f); }
}
