__aicore__ inline void VectorPadding_Mask2(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {
    (void)sharedTmpBuffer; (void)offset_ptr;
    const int32_t len = (data_length > 0) ? data_length : 1;
    const AscendC::UnaryRepeatParams p{1,1,8,8};
    if (2 == 1) {
        uint64_t bits = (len >= 64) ? UINT64_MAX : ((1ULL << (uint32_t)len) - 1ULL);
        uint64_t m[2] = {bits, 0ULL};
        AscendC::VectorPadding<float>(output, input0, (uint8_t)0, false, m, (uint8_t)1, p);
    } else {
        AscendC::VectorPadding<float>(output, input0, (uint8_t)0, false, (uint64_t)len, (uint8_t)1, p);
    }
}
