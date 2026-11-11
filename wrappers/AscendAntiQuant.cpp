__aicore__ inline void AscendAntiQuant(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {

    int32_t aq_off = (offset_ptr + 31) & ~31;
    offset_ptr = aq_off;
    AscendC::LocalTensor<int8_t> src_i8 = sharedTmpBuffer.ReinterpretCast<int8_t>()[offset_ptr / (int32_t)sizeof(int8_t)];
    src_i8.SetSize(data_length * (int32_t)sizeof(int8_t));
    offset_ptr += src_i8.GetSize();
    offset_ptr = (offset_ptr + 31) & ~31;
    AscendC::LocalTensor<half> dst_half = sharedTmpBuffer.ReinterpretCast<half>()[offset_ptr / (int32_t)sizeof(half)];
    dst_half.SetSize(data_length * (int32_t)sizeof(half));
    offset_ptr += dst_half.GetSize();
    for (int32_t i = 0; i < data_length; ++i) {
        src_i8.SetValue(i, static_cast<int8_t>(input0.GetValue(i)));
    }
    half scale = (half)0.1f;
    half offset_val = (half)0.0f;
    AscendC::AntiQuantShapeInfo shapeInfo;
    shapeInfo.scaleWidth = 1;
    shapeInfo.offsetWidth = 1;
    shapeInfo.scaleHeight = 1;
    shapeInfo.offsetHeight = 1;
    AscendC::AscendAntiQuant<int8_t, half, false>(dst_half, src_i8, offset_val, scale, sharedTmpBuffer, (uint32_t)data_length, shapeInfo);
    for (int32_t i = 0; i < data_length; ++i) {
        output.SetValue(i, (float)dst_half.GetValue(i));
    }
}
