__aicore__ inline void ProposalConcat(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const int32_t data_length) {

#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 100 || __CCE_AICORE__ == 200)
    int32_t repeat = data_length / 16;
    if (repeat > 255) repeat = 255;
    if (repeat < 0) repeat = 0;
    int32_t mode = 4;
    if (mode < 0) mode = 0;
    if (mode > 5) mode = 5;
    AscendC::ProposalConcat<float>(output, input0, repeat, mode);
#else
    uint32_t cnt = (data_length > 0) ? (uint32_t)data_length : 1U;
    AscendC::DataCopy(output, input0, cnt);
#endif
}
