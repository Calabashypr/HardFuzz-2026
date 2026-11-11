__aicore__ inline void WholeReduceMax(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const int32_t data_length) {

    int32_t mask = (data_length > 64) ? 64 : data_length;
    if (mask < 1) mask = 1;
    int32_t repeat_time = 1;
    int32_t dst_rep_stride = 1;
    int32_t src_blk_stride = 1;
    int32_t src_rep_stride = 8;
    AscendC::printf("[DBG] WholeReduceMax enter len=%d mask=%d in0=%f\n", data_length, mask, input0.GetValue(0));
    AscendC::WholeReduceMax<float>(output, input0, mask, repeat_time, dst_rep_stride, src_blk_stride, src_rep_stride);
    float v = output.GetValue(0);
    for (int32_t i = 0; i < data_length; ++i) {
        output.SetValue(i, v);
    }
    AscendC::printf("[DBG] WholeReduceMax exit out0=%f\n", output.GetValue(0));
}
