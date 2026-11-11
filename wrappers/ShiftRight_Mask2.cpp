
__aicore__ inline void ShiftRight_Mask2(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {
    int32_t off = (offset_ptr + 31) & ~31;
    offset_ptr = off;
    AscendC::LocalTensor<int32_t> src_int = sharedTmpBuffer.ReinterpretCast<int32_t>()[offset_ptr / (int32_t)sizeof(int32_t)];
    src_int.SetSize(64 * (int32_t)sizeof(int32_t));
    offset_ptr += src_int.GetSize();
    off = (offset_ptr + 31) & ~31;
    offset_ptr = off;
    AscendC::LocalTensor<int32_t> dst_int = sharedTmpBuffer.ReinterpretCast<int32_t>()[offset_ptr / (int32_t)sizeof(int32_t)];
    dst_int.SetSize(64 * (int32_t)sizeof(int32_t));
    offset_ptr += dst_int.GetSize();
    const AscendC::UnaryRepeatParams urp{1, 1, 8, 8};
    const uint64_t shift_amt = 2ULL;
    const int32_t len = (data_length > 0) ? data_length : 1;
    for (int32_t base = 0; base < len; base += 64) {
        int32_t chunk = len - base;
        if (chunk > 64) {
            chunk = 64;
        }
        for (int32_t i = 0; i < chunk; ++i) {
            src_int.SetValue(i, static_cast<int32_t>(input0.GetValue(base + i)));
        }
        const uint64_t mask = (uint64_t)chunk;
        AscendC::ShiftRight<int32_t>(dst_int, src_int, shift_amt, mask, (uint8_t)1, urp);
        for (int32_t i = 0; i < chunk; ++i) {
            const_cast<AscendC::LocalTensor<float> &>(output).SetValue(
                    base + i, static_cast<float>(dst_int.GetValue(i)));
        }
    }
}
