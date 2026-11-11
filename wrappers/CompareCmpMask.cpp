__aicore__ inline void CompareCmpMask(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const AscendC::LocalTensor<float> &input1,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {

    (void)sharedTmpBuffer;
    (void)offset_ptr;
    uint64_t mask = (uint64_t)((data_length > 64) ? 64 : data_length);
    if (mask < 1) mask = 1;
    AscendC::BinaryRepeatParams rp{ 1, 1, 1, 8, 8, 8 };
    AscendC::printf("[DBG] CompareCmpMask enter len=%d mask=%u a0=%f b0=%f\n", data_length, (uint32_t)mask, input0.GetValue(0), input1.GetValue(0));
    AscendC::Compare<float>(input0, input1, AscendC::CMPMODE::LT, mask, rp);

    for (int32_t i = 0; i < data_length; ++i) {
        output.SetValue(i, input0.GetValue(i));
    }
}
