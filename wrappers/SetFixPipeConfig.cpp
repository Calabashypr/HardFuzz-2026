__aicore__ inline void SetFixPipeConfig(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const int32_t data_length) {

    for (int32_t i = 0; i < data_length; ++i) {
        output.SetValue(i, input0.GetValue(i));
    }
}
