__aicore__ inline void Not(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {
    AscendC::LocalTensor<int16_t> input_for_logic, output_for_logic;
    int32_t not_offset = (offset_ptr + 31) & ~31;
    offset_ptr = not_offset;
    input_for_logic = sharedTmpBuffer.ReinterpretCast<int16_t>()[offset_ptr / (int32_t)sizeof(int16_t)];
    input_for_logic.SetSize(data_length * sizeof(int16_t));
    offset_ptr += input_for_logic.GetSize();
    not_offset = (offset_ptr + 31) & ~31;
    offset_ptr = not_offset;
    output_for_logic = sharedTmpBuffer.ReinterpretCast<int16_t>()[offset_ptr / (int32_t)sizeof(int16_t)];
    output_for_logic.SetSize(data_length * sizeof(int16_t));
    offset_ptr += output_for_logic.GetSize();
    for (int32_t i = 0; i < data_length; ++i) {
        input_for_logic.SetValue(i, static_cast<int16_t>(input0.GetValue(i)));
    }
    AscendC::Not(output_for_logic, input_for_logic, data_length);
    for (int32_t i = 0; i < data_length; ++i) {
        output.SetValue(i, static_cast<float>(output_for_logic.GetValue(i)));
    }
}
