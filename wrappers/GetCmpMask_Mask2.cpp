__aicore__ inline void GetCmpMask_Mask2(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {
    (void)input0;
    int32_t off = (offset_ptr + 15) & ~15; offset_ptr = off;
    AscendC::LocalTensor<uint8_t> dst = sharedTmpBuffer.ReinterpretCast<uint8_t>()[off];
    dst.SetSize(128); offset_ptr += 128;
    AscendC::GetCmpMask<uint8_t>(dst);
    for (int32_t i = 0; i < data_length; ++i) {
        uint8_t b = dst.GetValue(i >> 3);
        output.SetValue(i, ((b >> (i & 7)) & 0x1U) ? 1.0f : 0.0f);
    }
}
