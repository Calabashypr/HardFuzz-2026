__aicore__ inline void Duplicate(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const int32_t data_length) {

    uint32_t cnt = (data_length > 0) ? (uint32_t)data_length : 1U;
    AscendC::Duplicate<float>(output, input0.GetValue(0), cnt);
}
