
__aicore__ inline void LoadDataWithSparse_Mask2(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {
    // Copy the input.
    (void)sharedTmpBuffer;
    (void)offset_ptr;
    AscendC::DataCopy(output, input0, (uint32_t)data_length);
}
