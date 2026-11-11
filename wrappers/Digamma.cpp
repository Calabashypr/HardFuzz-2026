__aicore__ inline void Digamma(AscendC::LocalTensor<float> &dst,
        const AscendC::LocalTensor<float> &src,
        const AscendC::LocalTensor<float> & ,
        const int32_t data_length) {
    AscendC::Digamma(dst, src, (uint32_t)data_length);
}
