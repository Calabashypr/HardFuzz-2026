__aicore__ inline void Reciprocal(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {
    const float eps = 1e-3f;
    int32_t off = (offset_ptr + 31) & ~31;
    offset_ptr = off;
    AscendC::LocalTensor<float> safe_tensor = sharedTmpBuffer.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];
    safe_tensor.SetSize(data_length * (int32_t)sizeof(float));
    offset_ptr += safe_tensor.GetSize();

    for (int32_t i = 0; i < data_length; ++i) {
        float v = input0.GetValue(i);
        float s = (v >= eps || v <= -eps) ? v : eps;
        safe_tensor.SetValue(i, s);
    }
    AscendC::Reciprocal(output, safe_tensor, data_length);
}
