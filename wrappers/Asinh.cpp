__aicore__ inline void Asinh(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const int32_t data_length) {
    AscendC::Asinh(output, input0, (uint32_t)data_length);
}
