__aicore__ inline void Lgamma(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {

    int32_t lg_off = (offset_ptr + 31) & ~31;
    offset_ptr = lg_off;
    AscendC::LocalTensor<float> tmp_in = sharedTmpBuffer.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];
    tmp_in.SetSize(data_length * (int32_t)sizeof(float));
    offset_ptr += tmp_in.GetSize();

    for (int32_t i = 0; i < data_length; ++i) {
        float v = input0.GetValue(i);
        tmp_in.SetValue(i, (v > 0.5f) ? v : 0.5f);
    }

    AscendC::Lgamma(output, tmp_in, data_length);
}
