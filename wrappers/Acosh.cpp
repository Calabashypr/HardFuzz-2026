__aicore__ inline void Acosh(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {

    int32_t ac_off = (offset_ptr + 31) & ~31;
    offset_ptr = ac_off;
    AscendC::LocalTensor<float> tmp_clamped = sharedTmpBuffer.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];
    tmp_clamped.SetSize(data_length * (int32_t)sizeof(float));
    offset_ptr += tmp_clamped.GetSize();
    AscendC::Maxs(tmp_clamped, input0, (float)1.0f, data_length);

    AscendC::Acosh(output, tmp_clamped, data_length);
}
