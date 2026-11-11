__aicore__ inline void Fill(const AscendC::LocalTensor<float> &output,
        const int32_t data_length) {

    const float fillValue = 1.0f;
    uint32_t cnt = (data_length > 0) ? (uint32_t)data_length : 1U;
    AscendC::Duplicate<float>(output, fillValue, cnt);
}
