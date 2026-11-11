__aicore__ inline void LeakyRelu(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const int32_t data_length) {

    const float default_alpha = 0.01f;
    AscendC::LeakyRelu(output, input0, default_alpha, data_length);
}
