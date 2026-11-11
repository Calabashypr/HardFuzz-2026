__aicore__ inline void Fmod(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const AscendC::LocalTensor<float> &input1,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {
    const float eps = 1e-3f;
    int32_t off = (offset_ptr + 31) & ~31;
    offset_ptr = off;
    AscendC::LocalTensor<float> safe_den = sharedTmpBuffer.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];
    safe_den.SetSize(data_length * (int32_t)sizeof(float));
    offset_ptr += safe_den.GetSize();

    for (int32_t i = 0; i < data_length; ++i) {
        float b = input1.GetValue(i);
        float s = (b >= eps || b <= -eps) ? b : (b >= 0.0f ? eps : -eps);
        safe_den.SetValue(i, s);
    }
    AscendC::Fmod(output, input0, safe_den, data_length);
}
