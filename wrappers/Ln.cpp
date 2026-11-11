__aicore__ inline void Ln(const AscendC::LocalTensor<float> &output, const AscendC::LocalTensor<float> &input0, int &offset_ptr, const int32_t data_length) {
    AscendC::Abs(output, input0, data_length);
    AscendC::Mins(output, output, (float)2147483647*1.0f, data_length);
    AscendC::Maxs(output, output, (float)1e-3*1.0f, data_length);
    AscendC::Ln(output, output, data_length);
}

__aicore__ inline void Ln(const AscendC::LocalTensor<float> &output, int &offset_ptr, const int32_t data_length) {
    AscendC::Abs(output, output, data_length);
    AscendC::Mins(output, output, (float)2147483647*1.0f, data_length);
    AscendC::Maxs(output, output, (float)1e-3*1.0f, data_length);
    AscendC::Ln(output, output, data_length);
}
