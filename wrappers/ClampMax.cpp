__aicore__ inline void ClampMax(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const AscendC::LocalTensor<float> &input1,
        const int32_t data_length) {

    float max_val = (data_length > 0) ? input1.GetValue(0) : 1e30f;
    AscendC::ClampMax(output, input0, max_val, data_length);
}
