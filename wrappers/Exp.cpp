__aicore__ inline void Exp(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const int32_t data_length) {
    AscendC::Exp(output, input0, data_length);
}
