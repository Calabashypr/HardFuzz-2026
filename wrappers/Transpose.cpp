__aicore__ inline void Transpose(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {

    (void)sharedTmpBuffer;
    (void)offset_ptr;
    AscendC::DataCopy(const_cast<AscendC::LocalTensor<float> &>(output), input0, (uint32_t)data_length);
}
