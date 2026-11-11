__aicore__ inline void SetFmatrix(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const int32_t data_length) {

    float raw = input0.GetValue(0);

    int32_t rawI = (int32_t)raw;
    uint16_t l1H = (uint16_t)(rawI > 1 ? (rawI < 32767 ? (uint16_t)rawI : 32767U) : 1U);
    uint16_t l1W = (uint16_t)(data_length > 0 ? (data_length < 32767 ? (uint16_t)data_length : 32767U) : 1U);
    uint8_t padList[4] = {0, 0, 0, 0};
    AscendC::SetFmatrix(l1H, l1W, padList, AscendC::FmatrixMode::FMATRIX_LEFT);

    AscendC::DataCopy(output, input0, (uint32_t)data_length);
}
