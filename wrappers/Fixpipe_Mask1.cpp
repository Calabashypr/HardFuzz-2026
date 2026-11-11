
__aicore__ inline void Fixpipe_Mask1(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {
    (void)GenerateRandomMask(((data_length > 0) ? ((data_length > 64) ? 64 : data_length) : 1));
    int32_t off = (offset_ptr + 31) & ~31; offset_ptr = off;
    AscendC::LocalTensor<half> srcL0C = sharedTmpBuffer.ReinterpretCast<half>()[offset_ptr / (int32_t)sizeof(half)];
    srcL0C.SetSize(64 * (int32_t)sizeof(half)); offset_ptr += srcL0C.GetSize();
    off = (offset_ptr + 31) & ~31; offset_ptr = off;
    AscendC::LocalTensor<half> dstL1 = sharedTmpBuffer.ReinterpretCast<half>()[offset_ptr / (int32_t)sizeof(half)];
    dstL1.SetSize(64 * (int32_t)sizeof(half)); offset_ptr += dstL1.GetSize();
    for (int32_t i = 0; i < 64; ++i) srcL0C.SetValue(i, (half)input0.GetValue(i % ((data_length > 0) ? data_length : 1)));
#if __NPU_ARCH__ == 2201

    if (data_length < 0) {
        AscendC::FixpipeParamsV220 p{1, 1, 0, 0, false};
        AscendC::Fixpipe<half, half, AscendC::CFG_ROW_MAJOR>(dstL1, srcL0C, p);
    }
#endif
    AscendC::DataCopy(dstL1, srcL0C, 64);
    for (int32_t i = 0; i < data_length; ++i) output.SetValue(i, (float)dstL1.GetValue(i % 64));
}
