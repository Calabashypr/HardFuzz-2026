__aicore__ inline void MulAddDst(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const AscendC::LocalTensor<float> &input1,
        const int32_t data_length) {

    AscendC::LocalTensor<float> dstMutable = const_cast<AscendC::LocalTensor<float>&>(output);
    for (int32_t i = 0; i < data_length; ++i) {
        dstMutable.SetValue(i, 1.0f);
    }
    AscendC::MulAddDst(dstMutable, input0, input1, data_length);
}
