__aicore__ inline void Maxs(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const int32_t data_length) {

    const float default_scalar = 0.0f;
    AscendC::Maxs(output, input0, default_scalar, data_length);
}
