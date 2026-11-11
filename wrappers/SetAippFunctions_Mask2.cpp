
__aicore__ inline void SetAippFunctions_Mask2(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        GM_ADDR src0Gm,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {
    (void)sharedTmpBuffer; (void)offset_ptr;
    AscendC::GlobalTensor<uint8_t> src0;
    src0.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(src0Gm), 1024U);
    AscendC::AippParams<half> cfg{};
    cfg.cPaddingParams.cPaddingMode = 0;
    cfg.cPaddingParams.cPaddingValue = (half)0;
    cfg.singleLineParams.isSingleLineCopy = false;
    AscendC::SetAippFunctions<uint8_t, half>(src0, AscendC::AippInputFormat::YUV400_U8, cfg);
    AscendC::DataCopy(output, input0, (uint32_t)data_length);
}
