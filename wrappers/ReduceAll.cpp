__aicore__ inline void ReduceAll(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t sharedTmpBytes,
        const int32_t data_length) {

    for (int32_t i = 0; i < data_length; ++i) {
        output.SetValue(i, input0.GetValue(i));
    }
}
