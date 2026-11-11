__aicore__ inline void ScalarCast(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const int32_t calCount) {
    for (int32_t i = 0; i < calCount; ++i) {
        float v = input0.GetValue(i);
        int32_t t = AscendC::ScalarCast<float, int32_t, AscendC::RoundMode::CAST_ROUND>(v);
        float r = AscendC::ScalarCast<int32_t, float, AscendC::RoundMode::CAST_NONE>(t);
        output.SetValue(i, r);
    }
}
