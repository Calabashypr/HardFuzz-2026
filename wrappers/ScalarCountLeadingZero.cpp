__aicore__ inline void ScalarCountLeadingZero(const AscendC::LocalTensor<float> &inout,
        const int32_t data_length) {

    for (int32_t i = 0; i < data_length; ++i) {
        int32_t v = static_cast<int32_t>(inout.GetValue(i));
        uint64_t u = static_cast<uint64_t>(static_cast<int64_t>(v));
        int64_t lz = AscendC::ScalarCountLeadingZero(u);
        inout.SetValue(i, static_cast<float>(lz));
    }
}
