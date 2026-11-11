__aicore__ inline void DropOut(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const AscendC::LocalTensor<float> &input1,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {

    int32_t off = (offset_ptr + 31) & ~31;
    offset_ptr = off;

    AscendC::LocalTensor<float> xLocal = sharedTmpBuffer.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];
    xLocal.SetSize(data_length * (int32_t)sizeof(float));
    offset_ptr += xLocal.GetSize();
    offset_ptr = (offset_ptr + 31) & ~31;

    AscendC::LocalTensor<uint8_t> maskLocal = sharedTmpBuffer[offset_ptr];
    maskLocal.SetSize((uint32_t)data_length);
    offset_ptr += (int32_t)data_length;
    offset_ptr = (offset_ptr + 31) & ~31;

    AscendC::LocalTensor<uint8_t> dropTmp = sharedTmpBuffer[offset_ptr];
    dropTmp.SetSize(1024U);
    offset_ptr += 1024;
    offset_ptr = (offset_ptr + 31) & ~31;

    for (int32_t i = 0; i < data_length; ++i) {
        xLocal.SetValue(i, input0.GetValue(i));
        float m = input1.GetValue(i);
        maskLocal.SetValue(i, (uint8_t)(m > 0.5f ? 1 : 0));
    }

    AscendC::DropOutShapeInfo info;
    info.firstAxis = 1;
    info.srcLastAxis = (uint32_t)data_length;
    info.maskLastAxis = (uint32_t)data_length;
    const float probValue = 0.8f;

    AscendC::DropOut<float, false, 0>(output, xLocal, maskLocal, dropTmp, probValue, info);
}
