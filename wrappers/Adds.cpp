__aicore__ inline void Adds(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const int32_t data_length) {

    const float default_scalar = 1.0f;
    AscendC::Adds(output, input0, default_scalar, data_length);
}
