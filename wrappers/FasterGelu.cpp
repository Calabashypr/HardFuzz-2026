__aicore__ inline void FasterGelu(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const int32_t data_length) {

    AscendC::DataCopy(const_cast<AscendC::LocalTensor<float>&>(output), input0, (uint32_t)data_length);
}
