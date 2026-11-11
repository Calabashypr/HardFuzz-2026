__aicore__ inline void Cast(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const int32_t data_length) {
    for (int32_t i = 0; i < data_length; ++i) {
        float v = input0.GetValue(i);
        int32_t t = static_cast<int32_t>(v);
        output.SetValue(i, static_cast<float>(t));
    }
}
