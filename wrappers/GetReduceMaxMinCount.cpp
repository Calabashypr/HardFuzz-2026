__aicore__ inline void GetReduceMaxMinCount(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const int32_t data_length) {
    int32_t grm_mask = (data_length > 64) ? 64 : data_length;
    uint8_t grm_repeat = (uint8_t)((data_length + 63) / 64);
    uint8_t grm_dst_rep = 8;
    uint8_t grm_src_blk = 1;
    uint8_t grm_src_rep = 8;
    AscendC::WholeReduceMax<float>(const_cast<AscendC::LocalTensor<float>&>(output), input0, grm_mask, grm_repeat, grm_dst_rep, grm_src_blk, grm_src_rep);
    float grm_val = 0.0f;
    float grm_idx = 0.0f;
    AscendC::GetReduceMaxMinCount<float>(grm_val, grm_idx);

    for (int32_t i = 0; i < data_length; ++i) {
        output.SetValue(i, grm_val);
    }
    if (data_length >= 2) { output.SetValue(1, grm_idx); }
}
