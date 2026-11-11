__aicore__ inline void Abs(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const int32_t data_length) {
    AscendC::Abs(output, input0, data_length);
}
