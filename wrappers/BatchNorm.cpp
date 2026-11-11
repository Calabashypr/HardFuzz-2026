__aicore__ inline void BatchNorm(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &inputX,
        const AscendC::LocalTensor<float> &inputGamma,
        const AscendC::LocalTensor<float> &inputBeta,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {

    for (int32_t i = 0; i < data_length; ++i) {
        output.SetValue(i, inputX.GetValue(i));
    }
}
