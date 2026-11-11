__aicore__ inline void SetVectorMask_Mask1(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {
    (void)sharedTmpBuffer; (void)offset_ptr;
    const int32_t len = (data_length > 0) ? data_length : 1;
    if (1 == 1) {
        const uint64_t bits = GenerateRandomMask((len > 64) ? 64 : len);
        AscendC::SetVectorMask<float>((uint64_t)0, bits);
    } else {
        AscendC::SetVectorMask<float>(len);
    }
    AscendC::DataCopy(output, input0, (uint32_t)len);
}
