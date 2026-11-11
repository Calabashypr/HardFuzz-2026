__aicore__ inline void LoadDataWithTranspose(const AscendC::LocalTensor<float> &output,
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
        float v = (data_length > 0) ? input0.GetValue(i % data_length) : 0.0f;
        srcTile.SetValue(i, v);
    }
    AscendC::LoadData2dTransposeParams loadDataParams;
    loadDataParams.startIndex = 0;
    loadDataParams.repeatTimes = 1;
    loadDataParams.srcStride = 1;
    loadDataParams.dstGap = 0;
    loadDataParams.dstFracGap = 0;
    AscendC::LoadDataWithTranspose<float>(dstTile, srcTile, loadDataParams);

    AscendC::DataCopy(output, dstTile, (uint32_t)data_length);
}
