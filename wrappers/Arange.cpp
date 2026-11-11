__aicore__ inline void Arange(const AscendC::LocalTensor<float> &output,
        const int32_t data_length) {

    uint32_t count = (data_length > 0) ? (uint32_t)data_length : 1U;
#ifdef ASCENDC_HAS_ARANGE_API
    const float firstValue = 0.0f;
    const float diffValue = 1.0f;
    AscendC::Arange<float>(output, firstValue, diffValue, count);
#else
    AscendC::CreateVecIndex(output, 0.0f, count);
#endif
}
