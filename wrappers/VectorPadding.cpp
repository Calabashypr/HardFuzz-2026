__aicore__ inline void VectorPadding(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const int32_t data_length) {

    AscendC::VectorPadding(output, input0, (uint8_t)0, false, (uint32_t)data_length);
}
