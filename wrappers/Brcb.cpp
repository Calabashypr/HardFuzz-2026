__aicore__ inline void Brcb(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const int32_t data_length) {

    uint8_t repeat_time = (uint8_t)((data_length + 63) / 64);
    if (repeat_time < 1) repeat_time = 1;
    AscendC::BrcbRepeatParams brcb_params{};
    brcb_params.dstBlkStride = 1;
    brcb_params.dstRepStride = 8;
    AscendC::Brcb(output, input0, repeat_time, brcb_params);
}
