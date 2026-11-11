__aicore__ inline void Pad(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {

    constexpr int32_t PAD_COLS = 8;
    int32_t pad_dst_len = data_length + PAD_COLS;
    int32_t pad_off = (offset_ptr + 31) & ~31;
    offset_ptr = pad_off;
    AscendC::LocalTensor<float> pad_dst = sharedTmpBuffer.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];
    pad_dst.SetSize(pad_dst_len * (int32_t)sizeof(float));
    offset_ptr += pad_dst.GetSize();
    offset_ptr = (offset_ptr + 31) & ~31;
    for (int32_t i = 0; i < data_length; ++i) pad_dst.SetValue(i, input0.GetValue(i));
    for (int32_t i = data_length; i < pad_dst_len; ++i) pad_dst.SetValue(i, 0.0f);
    for (int32_t i = 0; i < data_length; ++i) output.SetValue(i, pad_dst.GetValue(i));
}
