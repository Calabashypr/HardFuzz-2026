__aicore__ inline void SetLoadDataBoundary(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const int32_t data_length) {

    float raw = input0.GetValue(0);
    int32_t rawI = (int32_t)raw;

    uint32_t boundaryValue = (rawI >= 1024) ? (uint32_t)rawI : 0U;
    AscendC::SetLoadDataBoundary(boundaryValue);

    AscendC::DataCopy(output, input0, (uint32_t)data_length);
}
