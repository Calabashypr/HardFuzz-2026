__aicore__ inline void SetMaskCount_Mask2(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {
    (void)sharedTmpBuffer; (void)offset_ptr;
    AscendC::SetMaskCount();
    AscendC::DataCopy(output, input0, (uint32_t)((data_length > 0) ? data_length : 1));
}
