__aicore__ inline void MulCast(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const AscendC::LocalTensor<float> &input1,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {

    int32_t mc_off = (offset_ptr + 31) & ~31;
    offset_ptr = mc_off;
    AscendC::LocalTensor<half> src0_half, src1_half;
    AscendC::LocalTensor<int8_t> dst_i8;
    src0_half = sharedTmpBuffer.ReinterpretCast<half>()[offset_ptr / (int32_t)sizeof(half)];
    src0_half.SetSize(data_length * (int32_t)sizeof(half));
    offset_ptr += src0_half.GetSize();
    offset_ptr = (offset_ptr + 31) & ~31;
    src1_half = sharedTmpBuffer.ReinterpretCast<half>()[offset_ptr / (int32_t)sizeof(half)];
    src1_half.SetSize(data_length * (int32_t)sizeof(half));
    offset_ptr += src1_half.GetSize();
    offset_ptr = (offset_ptr + 31) & ~31;
    dst_i8 = sharedTmpBuffer.ReinterpretCast<int8_t>()[offset_ptr / (int32_t)sizeof(int8_t)];
    dst_i8.SetSize(data_length * (int32_t)sizeof(int8_t));
    offset_ptr += dst_i8.GetSize();
    for (int32_t i = 0; i < data_length; ++i) {
        src0_half.SetValue(i, (half)input0.GetValue(i));
        src1_half.SetValue(i, (half)input1.GetValue(i));
    }
    AscendC::MulCast(dst_i8, src0_half, src1_half, (uint32_t)data_length);
    for (int32_t i = 0; i < data_length; ++i) {
        output.SetValue(i, (float)dst_i8.GetValue(i));
    }
}
