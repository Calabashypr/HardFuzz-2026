__aicore__ inline void ClampMin(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const float scalar_min,
        const int32_t data_length) {
    AscendC::ClampMin(output, input0, scalar_min, data_length);
}
