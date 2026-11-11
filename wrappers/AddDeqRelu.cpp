__aicore__ inline void AddDeqRelu(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const AscendC::LocalTensor<float> &input1,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {

    int32_t adr_off = (offset_ptr + 31) & ~31;
    offset_ptr = adr_off;
    AscendC::LocalTensor<int32_t> src0_i32, src1_i32;
    AscendC::LocalTensor<half> dst_half;
    src0_i32 = sharedTmpBuffer.ReinterpretCast<int32_t>()[offset_ptr / (int32_t)sizeof(int32_t)];
    src0_i32.SetSize(data_length * (int32_t)sizeof(int32_t));
    offset_ptr += src0_i32.GetSize();
    offset_ptr = (offset_ptr + 31) & ~31;
    src1_i32 = sharedTmpBuffer.ReinterpretCast<int32_t>()[offset_ptr / (int32_t)sizeof(int32_t)];
    src1_i32.SetSize(data_length * (int32_t)sizeof(int32_t));
    offset_ptr += src1_i32.GetSize();
    offset_ptr = (offset_ptr + 31) & ~31;
    dst_half = sharedTmpBuffer.ReinterpretCast<half>()[offset_ptr / (int32_t)sizeof(half)];
    dst_half.SetSize(data_length * (int32_t)sizeof(half));
    offset_ptr += dst_half.GetSize();
    for (int32_t i = 0; i < data_length; ++i) {
        src0_i32.SetValue(i, static_cast<int32_t>(input0.GetValue(i)));
        src1_i32.SetValue(i, static_cast<int32_t>(input1.GetValue(i)));
    }
    half deq_scale = (half)0.1f;
    AscendC::SetDeqScale(deq_scale);
    AscendC::AddDeqRelu(dst_half, src0_i32, src1_i32, data_length);
    for (int32_t i = 0; i < data_length; ++i) {
        output.SetValue(i, (float)dst_half.GetValue(i));
    }
}
