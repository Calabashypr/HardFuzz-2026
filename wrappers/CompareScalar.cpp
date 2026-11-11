__aicore__ inline void CompareScalar(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const AscendC::LocalTensor<float> &input1,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {

    float scalar_val = (data_length > 0) ? input1.GetValue(0) : 0.0f;

    int32_t elements_per_repeat = (data_length >= 64) ? 64 : ((data_length + 7) & ~7);
    if (elements_per_repeat == 0) elements_per_repeat = 8;
    int32_t repeat_times = (data_length + elements_per_repeat - 1) / elements_per_repeat;
    uint64_t mask = ((uint64_t)(repeat_times - 1) << 8) | (uint64_t)elements_per_repeat;

    int32_t bitmask_bytes = ((data_length + 7) / 8 + 31) & ~31;
    int32_t bitmask_off = (offset_ptr + 31) & ~31;
    offset_ptr = bitmask_off + bitmask_bytes;
    AscendC::LocalTensor<uint8_t> bitmask;
    bitmask = sharedTmpBuffer.ReinterpretCast<uint8_t>()[bitmask_off];
    bitmask.SetSize(bitmask_bytes);
    AscendC::CompareScalar<float, uint8_t>(bitmask, input0, scalar_val, AscendC::CMPMODE::LT, mask);

    for (int32_t i = 0; i < data_length; ++i) {
        uint8_t byte_val = bitmask.GetValue(i >> 3);
        uint8_t bit = (byte_val >> (uint8_t)(i & 7)) & (uint8_t)0x1;
        output.SetValue(i, (bit != 0U) ? 1.0f : 0.0f);
    }
}
