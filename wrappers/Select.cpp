__aicore__ inline void Select(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const AscendC::LocalTensor<float> &input1,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {

    int32_t sel_off = (offset_ptr + 31) & ~31;
    offset_ptr = sel_off;
    AscendC::LocalTensor<uint8_t> sel_bits = sharedTmpBuffer[offset_ptr];
    sel_bits.SetSize(32);
    offset_ptr += sel_bits.GetSize();
    uint64_t mask = (uint64_t)((data_length > 64) ? 64 : data_length);
    if (mask < 1) mask = 1;
    uint8_t repeat_time = 1;
    AscendC::BinaryRepeatParams rp{ 1, 1, 1, 8, 8, 8 };
    AscendC::Compare<float, uint8_t>(sel_bits, input0, input1, AscendC::CMPMODE::LT, mask, repeat_time, rp);
    AscendC::Select<float, uint8_t>(output, sel_bits, input0, input1, AscendC::SELMODE::VSEL_CMPMASK_SPR, (uint32_t)data_length);
}
