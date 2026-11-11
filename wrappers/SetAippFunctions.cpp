__aicore__ inline void SetAippFunctions(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        GM_ADDR src0Gm,
        const int32_t data_length) {

    AscendC::GlobalTensor<uint8_t> src0;
    src0.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(src0Gm), 1024U);
    AscendC::AippParams<half> aippConfig{};
    aippConfig.cPaddingParams.cPaddingMode = 0;
    aippConfig.cPaddingParams.cPaddingValue = (half)0;
    aippConfig.singleLineParams.isSingleLineCopy = false;
    AscendC::SetAippFunctions<uint8_t, half>(src0, AscendC::AippInputFormat::YUV400_U8, aippConfig);

    AscendC::DataCopy(output, input0, (uint32_t)data_length);
}
