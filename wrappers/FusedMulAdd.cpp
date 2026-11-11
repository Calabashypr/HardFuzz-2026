__aicore__ inline void FusedMulAdd(AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const AscendC::LocalTensor<float> &input1,
        const int32_t data_length) {
    for (int32_t i = 0; i < data_length; ++i) {
        output.SetValue(i, 1.0f);
    }
    AscendC::FusedMulAdd(output, input0, input1, data_length);
}
