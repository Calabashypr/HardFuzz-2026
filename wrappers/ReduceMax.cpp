__aicore__ inline void ReduceMax(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t sharedTmpBytes,
        const int32_t data_length) {
    constexpr int32_t kReduceTmpBytes = 8192;
    int32_t off = (offset_ptr + 31) & ~31;
    int32_t off2 = (off + (int32_t)sizeof(float) + 31) & ~31;
    int32_t remain = sharedTmpBytes - off2;
    if (remain < kReduceTmpBytes || data_length <= 0) return;
    offset_ptr = off;
    AscendC::LocalTensor<float> dst_reduce = sharedTmpBuffer.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];
    dst_reduce.SetSize(1 * (int32_t)sizeof(float));
    offset_ptr += dst_reduce.GetSize();
    offset_ptr = (offset_ptr + 31) & ~31;
    AscendC::LocalTensor<float> tmp_reduce = sharedTmpBuffer.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];
    tmp_reduce.SetSize(kReduceTmpBytes);
    offset_ptr += kReduceTmpBytes;
    AscendC::ReduceMax<float>(dst_reduce, input0, tmp_reduce, data_length, 1, data_length);
    float v = dst_reduce.GetValue(0);
    for (int32_t i = 0; i < data_length; ++i) { output.SetValue(i, v); }
}
