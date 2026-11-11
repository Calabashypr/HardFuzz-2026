__aicore__ inline void Asin(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {

    int32_t ta_offset = (offset_ptr + 31) & ~31;
    offset_ptr = ta_offset;
    AscendC::LocalTensor<float> tmp_float = sharedTmpBuffer.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];
    tmp_float.SetSize(data_length * (int32_t)sizeof(float));
    offset_ptr += tmp_float.GetSize();
    AscendC::Tanh(tmp_float, input0, data_length);
    AscendC::Asin(output, tmp_float, data_length);
}
