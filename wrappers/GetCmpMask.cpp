__aicore__ inline void GetCmpMask(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {

    (void)input0;
    int32_t gcm_off = (offset_ptr + 15) & ~15;
    offset_ptr = gcm_off;
    AscendC::LocalTensor<uint8_t> dstLocal = sharedTmpBuffer.ReinterpretCast<uint8_t>()[gcm_off];
    dstLocal.SetSize(128);
    offset_ptr = gcm_off + 128;
    AscendC::printf("[DBG] GetCmpMask enter len=%d off=%d\n", data_length, gcm_off);
    AscendC::GetCmpMask<uint8_t>(dstLocal);

    for (int32_t i = 0; i < data_length; ++i) {
        uint8_t b = dstLocal.GetValue(i >> 3);
        uint8_t bit = (b >> (i & 7)) & 0x1;
        output.SetValue(i, (bit != 0U) ? 1.0f : 0.0f);
    }
}
