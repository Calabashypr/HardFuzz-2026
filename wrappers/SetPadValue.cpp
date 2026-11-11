__aicore__ inline void SetPadValue(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const int32_t data_length) {

    float paddingValue = input0.GetValue(0);
    AscendC::SetPadValue<float>(paddingValue);

    AscendC::DataCopy(output, input0, (uint32_t)data_length);
}
