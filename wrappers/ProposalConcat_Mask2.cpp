__aicore__ inline void ProposalConcat_Mask2(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {
    (void)sharedTmpBuffer; (void)offset_ptr;
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 100 || __CCE_AICORE__ == 200)
    int32_t repeat = data_length / 16;
    if (repeat < 0) repeat = 0;
    if (repeat > 255) repeat = 255;
    AscendC::ProposalConcat<float>(output, input0, repeat, 4);
#else
    AscendC::DataCopy(output, input0, (uint32_t)((data_length > 0) ? data_length : 1));
#endif
}
