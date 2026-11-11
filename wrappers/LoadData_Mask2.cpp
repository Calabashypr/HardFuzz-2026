
__aicore__ inline void LoadData_Mask2(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {
    int32_t off = (offset_ptr + 31) & ~31;
    offset_ptr = off;
    AscendC::LocalTensor<float> srcTile;
    srcTile.SetAddrWithOffset(sharedTmpBuffer, offset_ptr);
    srcTile.SetSize(256 * (int32_t)sizeof(float));
    offset_ptr += srcTile.GetSize();
    off = (offset_ptr + 511) & ~511;
    offset_ptr = off;
    AscendC::LocalTensor<float> dstTile;
    dstTile.SetAddrWithOffset(sharedTmpBuffer, offset_ptr);
    dstTile.SetSize(256 * (int32_t)sizeof(float));
    offset_ptr += dstTile.GetSize();
    for (int32_t i = 0; i < 256; ++i) {
        srcTile.SetValue(i, input0.GetValue(i % ((data_length > 0) ? data_length : 1)));
    }
    AscendC::LoadData2DParams p; p.startIndex = 0; p.repeatTimes = 1; p.srcStride = 1; p.sid = 0; p.dstGap = 0; p.ifTranspose = false;
    AscendC::LoadData<float>(dstTile, srcTile, p);
    AscendC::DataCopy(output, dstTile, (uint32_t)data_length);
}
