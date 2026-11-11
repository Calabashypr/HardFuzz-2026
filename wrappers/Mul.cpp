__aicore__ inline void Mul(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const AscendC::LocalTensor<float> &input1,
        const int32_t data_length) {
    AscendC::Mul(output, input0, input1, data_length);
}
