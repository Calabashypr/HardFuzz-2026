__aicore__ inline void InitConstValue(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const int32_t data_length) {

    float initValue = input0.GetValue(0);
    uint32_t bytes = (uint32_t)data_length * (uint32_t)sizeof(float);
    uint16_t blockNum = (uint16_t)((bytes + 31U) / 32U);
    if (blockNum == 0) blockNum = 1;
    AscendC::InitConstValue(output, {1, blockNum, 0, initValue});
}
