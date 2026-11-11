__aicore__ inline void CreateVecIndex_Mask1(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {
    (void)input0; (void)sharedTmpBuffer; (void)offset_ptr;
    const int32_t len = (data_length > 0) ? data_length : 1;
    if (1 == 1) {
        uint64_t bits = GenerateRandomMask((len > 64) ? 64 : len);
        uint64_t m[2] = {bits, 0ULL};
        AscendC::CreateVecIndex<float>(const_cast<AscendC::LocalTensor<float>&>(output), 0.0f, m, (uint8_t)1, (uint16_t)1, (uint8_t)8);
    } else {
        AscendC::CreateVecIndex<float>(const_cast<AscendC::LocalTensor<float>&>(output), 0.0f, (uint64_t)len, (uint8_t)1, (uint16_t)1, (uint8_t)8);
    }
}
