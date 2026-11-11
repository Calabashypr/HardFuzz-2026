__aicore__ inline void AscendQuant(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const AscendC::LocalTensor<float> &input1,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {

    float scale = 0.1f;
    float offset = 0.0f;
    for (int32_t i = 0; i < data_length; ++i) {
        float x = input0.GetValue(i);

        float scaled = scale * x + offset;

        float rounded = (scaled >= 0.0f) ? (scaled + 0.5f) : (scaled - 0.5f);

        int32_t clamped = (int32_t)rounded;
        if (clamped > 127) clamped = 127;
        if (clamped < -128) clamped = -128;
        output.SetValue(i, (float)clamped);
    }
}
