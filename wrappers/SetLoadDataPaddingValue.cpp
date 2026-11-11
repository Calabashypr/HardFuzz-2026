__aicore__ inline void SetLoadDataPaddingValue(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const int32_t data_length) {

    float padValue = input0.GetValue(0);
    AscendC::SetLoadDataPaddingValue<float>(padValue);

    AscendC::DataCopy(output, input0, (uint32_t)data_length);
}
