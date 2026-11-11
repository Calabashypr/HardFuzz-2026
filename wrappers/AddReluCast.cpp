__aicore__ inline void AddReluCast(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const AscendC::LocalTensor<float> &input1,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {

    int32_t arc_off = (offset_ptr + 31) & ~31;
    offset_ptr = arc_off;
    AscendC::LocalTensor<half> tmp_half = sharedTmpBuffer.ReinterpretCast<half>()[offset_ptr / (int32_t)sizeof(half)];
    tmp_half.SetSize(data_length * (int32_t)sizeof(half));
    offset_ptr += tmp_half.GetSize();
    AscendC::AddReluCast(tmp_half, input0, input1, data_length);
    for (int32_t i = 0; i < data_length; ++i) {
        output.SetValue(i, (float)tmp_half.GetValue(i));
    }
}
