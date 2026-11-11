__aicore__ inline void ScalarGetSFFValue(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const int32_t data_length) {

    for (int32_t i = 0; i < data_length; ++i) {
        uint64_t u64 = static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(input0.GetValue(i))));
        int64_t pos = AscendC::ScalarGetSFFValue<1>(u64);
        output.SetValue(i, static_cast<float>(pos));
    }
}
