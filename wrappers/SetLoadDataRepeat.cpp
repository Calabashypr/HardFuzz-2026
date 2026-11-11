__aicore__ inline void SetLoadDataRepeat(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const int32_t data_length) {

    AscendC::LoadDataRepeatParam repeatParams;

    repeatParams.repeatTime = 1;
    repeatParams.repeatStride = 0;
    repeatParams.repeatMode = 0;
    AscendC::SetLoadDataRepeat(repeatParams);

    AscendC::DataCopy(output, input0, (uint32_t)data_length);
}
