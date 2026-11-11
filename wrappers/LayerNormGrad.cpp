__aicore__ inline void LayerNormGrad(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &inputDy,
        const AscendC::LocalTensor<float> &inputX,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {

    for (int32_t i = 0; i < data_length; ++i) {
        output.SetValue(i, inputDy.GetValue(i));
    }
}
