__aicore__ inline void ConstOne(const AscendC::LocalTensor<float> &output,
        const int32_t data_length) {

    for (int32_t i = 0; i < data_length; ++i) {
        output.SetValue(i, 1.0f);
    }
}
