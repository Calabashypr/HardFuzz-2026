__aicore__ inline void Xor(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const AscendC::LocalTensor<float> &input1,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {
    AscendC::LocalTensor<int16_t> input_x, input_y, output_xy;
    int32_t xor_offset = (offset_ptr + 31) & ~31;
    offset_ptr = xor_offset;
    input_x = sharedTmpBuffer.ReinterpretCast<int16_t>()[offset_ptr / (int32_t)sizeof(int16_t)];
    input_x.SetSize(data_length * sizeof(int16_t));
    offset_ptr += input_x.GetSize();
    xor_offset = (offset_ptr + 31) & ~31;
    offset_ptr = xor_offset;
    input_y = sharedTmpBuffer.ReinterpretCast<int16_t>()[offset_ptr / (int32_t)sizeof(int16_t)];
    input_y.SetSize(data_length * sizeof(int16_t));
    offset_ptr += input_y.GetSize();
    xor_offset = (offset_ptr + 31) & ~31;
    offset_ptr = xor_offset;
    output_xy = sharedTmpBuffer.ReinterpretCast<int16_t>()[offset_ptr / (int32_t)sizeof(int16_t)];
    output_xy.SetSize(data_length * sizeof(int16_t));
    offset_ptr += output_xy.GetSize();
    for (int32_t i = 0; i < data_length; ++i) {
        input_x.SetValue(i, static_cast<int16_t>(input0.GetValue(i)));
        input_y.SetValue(i, static_cast<int16_t>(input1.GetValue(i)));
    }
    AscendC::Xor(output_xy, input_x, input_y, data_length);
    for (int32_t i = 0; i < data_length; ++i) {
        output.SetValue(i, static_cast<float>(output_xy.GetValue(i)));
    }
}
