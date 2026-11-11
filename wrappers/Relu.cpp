__aicore__ inline void Relu(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const int32_t data_length) {
    AscendC::Relu(output, input0, data_length);
}
