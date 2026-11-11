__aicore__ inline void GeGLU(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const AscendC::LocalTensor<float> &input1,
        const int32_t data_length) {

    AscendC::GeGLU<float, false>(const_cast<AscendC::LocalTensor<float>&>(output), input0, input1, (uint32_t)data_length);
}
