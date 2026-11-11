__aicore__ inline void GetAccVal_Mask1(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {
    (void)GenerateRandomMask(((data_length > 0) ? ((data_length > 64) ? 64 : data_length) : 1));
    (void)input0; (void)sharedTmpBuffer; (void)offset_ptr;
    const float acc = AscendC::GetAccVal<float>();
    for (int32_t i = 0; i < data_length; ++i) output.SetValue(i, acc);
}
