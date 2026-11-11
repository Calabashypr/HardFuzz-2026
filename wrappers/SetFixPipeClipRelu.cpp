__aicore__ inline void SetFixPipeClipRelu(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const int32_t data_length) {

    float raw = input0.GetValue(0);

    int32_t rawI = (int32_t)raw;
    uint64_t absU = (rawI >= 0) ? (uint64_t)rawI : (uint64_t)(-(int64_t)rawI);

    uint64_t clipReluMaxVal = (absU % 0xFFFFULL) + 1ULL;

    AscendC::DataCopy(output, input0, (uint32_t)data_length);
}
