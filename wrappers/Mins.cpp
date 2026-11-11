__aicore__ inline void Mins(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const AscendC::LocalTensor<float> &input1,
        const int32_t data_length) {

    float scalar_val = (data_length > 0) ? input1.GetValue(0) : 0.0f;
    AscendC::Mins(output, input0, scalar_val, data_length);
}
