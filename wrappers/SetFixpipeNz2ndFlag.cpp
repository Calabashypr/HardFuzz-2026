__aicore__ inline void SetFixpipeNz2ndFlag(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const int32_t data_length) {

    float raw = input0.GetValue(0);

    int32_t rawI = (int32_t)raw;
    uint32_t absU = (rawI >= 0) ? (uint32_t)rawI : (uint32_t)(-rawI);
    uint16_t ndNum = (uint16_t)((absU % 65535U) + 1U);
    uint16_t srcNdStride = (uint16_t)((absU % 512U) + 1U);
    uint16_t dstNdStride = (uint16_t)((absU % 65535U) + 1U);
    AscendC::SetFixpipeNz2ndFlag(ndNum, srcNdStride, dstNdStride);

    AscendC::DataCopy(output, input0, (uint32_t)data_length);
}
