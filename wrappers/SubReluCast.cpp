__aicore__ inline void SubReluCast(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const AscendC::LocalTensor<float> &input1,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {

    int32_t src_off = (offset_ptr + 31) & ~31;
    offset_ptr = src_off;
    AscendC::LocalTensor<half> dst_half = sharedTmpBuffer.ReinterpretCast<half>()[offset_ptr / (int32_t)sizeof(half)];
    dst_half.SetSize(data_length * (int32_t)sizeof(half));
    offset_ptr += dst_half.GetSize();
    AscendC::SubReluCast(dst_half, input0, input1, (uint32_t)data_length);
    for (int32_t i = 0; i < data_length; ++i) {
        output.SetValue(i, (float)dst_half.GetValue(i));
    }
}
