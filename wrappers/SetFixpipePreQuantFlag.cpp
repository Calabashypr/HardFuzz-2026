__aicore__ inline void SetFixpipePreQuantFlag(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const int32_t data_length) {

    float raw = input0.GetValue(0);

    int32_t rawI = (int32_t)raw;
    uint64_t config = (rawI >= 0) ? (uint64_t)rawI : (uint64_t)(-(int64_t)rawI);
    AscendC::SetFixpipePreQuantFlag(config);

    AscendC::DataCopy(output, input0, (uint32_t)data_length);
}
