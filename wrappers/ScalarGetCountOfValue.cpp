__aicore__ inline void ScalarGetCountOfValue(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const int32_t data_length) {
    float target = 0.0f;
    int32_t count = 0;
    for (int32_t i = 0; i < data_length; ++i) {
        if (input0.GetValue(i) == target) count++;
    }
    for (int32_t i = 0; i < data_length; ++i) {
        output.SetValue(i, static_cast<float>(count));
    }
}
