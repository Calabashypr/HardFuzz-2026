__aicore__ inline void ReGlu(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const AscendC::LocalTensor<float> &input1,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {

    int32_t off = (offset_ptr + 31) & ~31;
    offset_ptr = off;
    AscendC::LocalTensor<uint8_t> tmpLocal = sharedTmpBuffer[offset_ptr];

    tmpLocal.SetSize((uint32_t)data_length * 3U * (uint32_t)sizeof(float));
    offset_ptr += tmpLocal.GetSize();
    AscendC::ReGlu<float, false>(
        const_cast<AscendC::LocalTensor<float>&>(output),
        input0,
        input1,
        tmpLocal,
        (uint32_t)data_length);
}
