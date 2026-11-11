__aicore__ inline void DataCopyPad(const AscendC::LocalTensor<float> &output,
        const AscendC::GlobalTensor<float> &srcGm,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {

    (void)sharedTmpBuffer; (void)offset_ptr;

    uint32_t copyCount = (uint32_t)data_length;
    if (copyCount > 20U) copyCount = 20U;

    AscendC::DataCopyExtParams copyParams{1, copyCount * (uint32_t)sizeof(float), 0U, 0U, 0U};
    AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
    AscendC::DataCopyPad(output, srcGm, copyParams, padParams);
}
