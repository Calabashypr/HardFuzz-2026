__aicore__ inline void Swish(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const int32_t data_length) {

    const float scalarValue = 1.702f;
    AscendC::Swish<float, false>(output, input0, (uint32_t)data_length, scalarValue);
}
