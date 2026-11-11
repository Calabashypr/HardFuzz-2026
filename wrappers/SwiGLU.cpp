__aicore__ inline void SwiGLU(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const AscendC::LocalTensor<float> &input1,
        const int32_t data_length) {

    const float scalarValue = 1.0f;
    AscendC::SwiGLU<float, false>(const_cast<AscendC::LocalTensor<float>&>(output), input0, input1, scalarValue, (uint32_t)data_length);
}
