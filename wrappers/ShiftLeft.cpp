__aicore__ inline void ShiftLeft(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {

    int32_t sl_off = (offset_ptr + 31) & ~31;
    offset_ptr = sl_off;
    AscendC::LocalTensor<int32_t> src_int, dst_int;
    src_int = sharedTmpBuffer.ReinterpretCast<int32_t>()[offset_ptr / (int32_t)sizeof(int32_t)];
    src_int.SetSize(data_length * (int32_t)sizeof(int32_t));
    offset_ptr += src_int.GetSize();
    sl_off = (offset_ptr + 31) & ~31;
    offset_ptr = sl_off;
    dst_int = sharedTmpBuffer.ReinterpretCast<int32_t>()[offset_ptr / (int32_t)sizeof(int32_t)];
    dst_int.SetSize(data_length * (int32_t)sizeof(int32_t));
    offset_ptr += dst_int.GetSize();
    for (int32_t i = 0; i < data_length; ++i) {
        src_int.SetValue(i, static_cast<int32_t>(input0.GetValue(i)));
    }
    const int32_t shift_amt = 2;
    AscendC::ShiftLeft(dst_int, src_int, shift_amt, data_length);
    for (int32_t i = 0; i < data_length; ++i) {
        output.SetValue(i, static_cast<float>(dst_int.GetValue(i)));
    }
}
