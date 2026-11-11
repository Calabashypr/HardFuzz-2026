__aicore__ inline void GetReduceMaxMinCount_Mask1(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {
    (void)GenerateRandomMask(((data_length > 0) ? ((data_length > 64) ? 64 : data_length) : 1));
    (void)sharedTmpBuffer; (void)offset_ptr;
    int32_t mask = (data_length > 64) ? 64 : data_length;
    if (mask < 1) mask = 1;
    uint8_t repeat = (uint8_t)(((data_length > 0 ? data_length : 1) + 63) / 64);
    AscendC::WholeReduceMax<float>(const_cast<AscendC::LocalTensor<float>&>(output), input0, mask, repeat, 8, 1, 8);
    float v = 0.0f, idx = 0.0f;
    AscendC::GetReduceMaxMinCount<float>(v, idx);
    for (int32_t i = 0; i < data_length; ++i) output.SetValue(i, v);
    if (data_length > 1) output.SetValue(1, idx);
}
