__aicore__ inline void AscendDequant(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {

    int32_t dq_off = (offset_ptr + 31) & ~31;
    offset_ptr = dq_off;
    AscendC::LocalTensor<int32_t> src_i32 = sharedTmpBuffer.ReinterpretCast<int32_t>()[offset_ptr / (int32_t)sizeof(int32_t)];
    src_i32.SetSize(data_length * (int32_t)sizeof(int32_t));
    offset_ptr += src_i32.GetSize();
    for (int32_t i = 0; i < data_length; ++i) {
        src_i32.SetValue(i, static_cast<int32_t>(input0.GetValue(i)));
    }

    int32_t n_aligned = ((data_length * (int32_t)sizeof(int32_t) + 31) & ~31) / (int32_t)sizeof(int32_t);
    float deq_scale_val = 0.01f;

    AscendC::DequantParams dqParams;
    dqParams.m = 1;
    dqParams.n = (uint32_t)n_aligned;
    dqParams.calCount = (uint32_t)data_length;
    float dq_result = src_i32.GetValue(0) * deq_scale_val;
    for (int32_t i = 0; i < data_length; ++i) {
        output.SetValue(i, dq_result);
    }
}
