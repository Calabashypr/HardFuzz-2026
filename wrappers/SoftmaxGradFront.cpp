__aicore__ inline void SoftmaxGradFront(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &gradTensor,
        const AscendC::LocalTensor<float> &srcTensor,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {

    for (int32_t i = 0; i < data_length; ++i) {
        output.SetValue(i, gradTensor.GetValue(i));
    }
}
