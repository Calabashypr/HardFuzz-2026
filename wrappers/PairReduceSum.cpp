__aicore__ inline void PairReduceSum(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const int32_t data_length) {

    int32_t mask = (data_length > 64) ? 64 : data_length;
    if (mask < 1) mask = 1;
    int32_t repeat_time = 1;
    int32_t dst_rep_stride = 1;
    int32_t src_blk_stride = 1;
    int32_t src_rep_stride = 8;
    AscendC::printf("[DBG] PairReduceSum enter len=%d mask=%d in0=%f\n", data_length, mask, input0.GetValue(0));
    AscendC::PairReduceSum<float>(output, input0, repeat_time, mask, dst_rep_stride, src_blk_stride, src_rep_stride);
    AscendC::printf("[DBG] PairReduceSum exit out0=%f\n", output.GetValue(0));
}
