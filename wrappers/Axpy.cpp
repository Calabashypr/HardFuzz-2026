__aicore__ inline void Axpy(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const AscendC::LocalTensor<float> &input1,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {

    AscendC::DataCopy(output, input1, (uint32_t)data_length);

    const float alpha = 2.0f;
    AscendC::Axpy(output, input0, alpha, data_length);
}
