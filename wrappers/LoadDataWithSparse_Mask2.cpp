
__aicore__ inline void LoadDataWithSparse_Mask2(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {
    int32_t off = (offset_ptr + 31) & ~31; offset_ptr = off;
    AscendC::LocalTensor<int8_t> srcB1; srcB1.SetAddrWithOffset(sharedTmpBuffer, offset_ptr); srcB1.SetSize(512); offset_ptr += 512;
    off = (offset_ptr + 31) & ~31; offset_ptr = off;
    AscendC::LocalTensor<uint8_t> idxB1; idxB1.SetAddrWithOffset(sharedTmpBuffer, offset_ptr); idxB1.SetSize(128); offset_ptr += 128;
    off = (offset_ptr + 511) & ~511; offset_ptr = off;
    AscendC::LocalTensor<int8_t> dstB2; dstB2.SetAddrWithOffset(sharedTmpBuffer, offset_ptr); dstB2.SetSize(512); offset_ptr += 512;
    for (int32_t i = 0; i < 512; ++i) {
        float v = input0.GetValue(i % ((data_length > 0) ? data_length : 1));
        srcB1.SetValue(i, (int8_t)(v > 127.0f ? 127 : v < -128.0f ? -128 : (int8_t)v));
    }
    for (int32_t i = 0; i < 128; ++i) idxB1.SetValue(i, (uint8_t)0);
    AscendC::LoadData2DParams p; p.repeatTimes = 1; p.srcStride = 0; p.ifTranspose = false;
    AscendC::LoadDataWithSparse(dstB2, srcB1, idxB1, p);
    AscendC::DataCopy(output, input0, (uint32_t)data_length);
}
