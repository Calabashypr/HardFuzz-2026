__aicore__ inline void Compare(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const AscendC::LocalTensor<float> &input1,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {

    int32_t cp_off = (offset_ptr + 31) & ~31;
    offset_ptr = cp_off;
    AscendC::LocalTensor<uint8_t> dst_bits = sharedTmpBuffer[offset_ptr];
    dst_bits.SetSize(32);
    offset_ptr += dst_bits.GetSize();
    uint64_t mask = (uint64_t)((data_length > 64) ? 64 : data_length);
    if (mask < 1) mask = 1;
    uint8_t repeat_time = 1;
    AscendC::BinaryRepeatParams rp{ 1, 1, 1, 8, 8, 8 };
    AscendC::printf("[DBG] Compare enter len=%d mask=%u a0=%f b0=%f\n", data_length, (uint32_t)mask, input0.GetValue(0), input1.GetValue(0));
    AscendC::Compare<float, uint8_t>(dst_bits, input0, input1, AscendC::CMPMODE::LT, mask, repeat_time, rp);
    for (int32_t i = 0; i < data_length; ++i) {
        uint8_t b = dst_bits.GetValue(i >> 3);
        uint8_t bit = (b >> (i & 7)) & 0x1;
        output.SetValue(i, (bit != 0U) ? 1.0f : 0.0f);
    }
    AscendC::printf("[DBG] Compare exit bits0=%u out0=%f\n", (uint32_t)dst_bits.GetValue(0), output.GetValue(0));
}
