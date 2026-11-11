__aicore__ inline void Muls(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const int32_t data_length) {

    const float default_scalar = 2.0f;
    AscendC::Muls(output, input0, default_scalar, data_length);
}
