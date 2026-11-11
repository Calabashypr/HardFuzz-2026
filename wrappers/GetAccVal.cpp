__aicore__ inline void GetAccVal(const AscendC::LocalTensor<float> &output,
        const int32_t data_length) {

    float acc = AscendC::GetAccVal<float>();
    for (int32_t i = 0; i < data_length; ++i) {
        output.SetValue(i, acc);
    }
}
